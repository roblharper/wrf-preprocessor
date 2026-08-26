# WRF-PINN pre-processor

Consolidates heterogeneous atmospheric NetCDF data into **one normalized binary
per HRRR snapshot** that the PINN reads directly. It lives *outside* the model
codebase on purpose: the PINN should never become a data reader. All the messy
differences between sources (variables, units, coordinates, resolutions) are
resolved here, once, so the model always sees exactly one format.

## What it produces

For each HRRR snapshot it writes one `.npy` of rows in a fixed canonical schema:

```
x, y, z, t, u, v, w, theta, p_prime, source
```

plus a shared `metadata.json` (schema, the global normalization recipe, the
source-code map, the train/test split, and the case definition). The cases are
written into `train/` and `test/` subfolders (see below).

`x, y, z, t` are **required**; a row missing any of them is dropped. The physical
columns `u, v, w, theta, p_prime` are **optional**: a source that does not
measure one leaves it `NaN` (the row is kept and the gap is logged). `source` is
a per-row integer tag identifying the data category the row came from, so the
model can weight the loss per source.

## What a "case" is

A case is **one HRRR snapshot**. HRRR is the **anchor condition**, not a data
source: it defines each case's time window and x/y bounding box but contributes
no rows of its own. The rows in a case come from the LES and sensor data that
belong to that snapshot. "Belong to it" means:

- **time:** within a tolerance window of the snapshot time
  (`TIME_TOLERANCE_SECONDS`, default ±30 min);
- **space:** inside the snapshot's x/y bounding box.

Rows matching no snapshot are **dropped**, and a snapshot with no matching rows
is skipped. Normalization is **global**: one recipe fit over all synced rows,
applied to every case, so they share a scale.

The two data categories in the `source` column are:

| Code | Category | Sources |
|------|----------|---------|
| `0` | simulation | LASSO WRF-LES (`wrfout`) |
| `1` | sensor | ground-observation streams (ecor, smos, twr, ...) |

### Train / test split

Cases are split **by whole case** (seeded, reproducible) into `train/` and
`test/` subfolders, so the held-out set is genuinely unseen. Controlled by
`--test-fraction` (default 0.2) and `--seed` (default 0).

## Pipeline

```
NetCDF sources ─▶ reader ─▶ (derive) ─▶ sync ─▶ normalize ─▶ writer ─▶ <snapshot>.npy
                                                                        + metadata.json
```

| Stage | File | Job |
|-------|------|-----|
| **Reader** | `reader.py` | Stream any NetCDF in chunks (memory-safe); map raw variables to canonical columns by dimension name. |
| **Processor** | `processor.py` | Run a source's `derive` hook (computed columns), assemble canonical rows, then normalize. |
| **Sync** | `sync.py` | Build one snapshot per HRRR time (its time + x/y bbox); attach LES/sensor rows by time window + bbox; drop the rest. |
| **Writer** | `writer.py` | Split cases into `train/` and `test/`, write each `.npy` + a shared `metadata.json`. |
| **Orchestrator** | `orchestrator.py` | Thin wiring of the above + CLI. |
| **Registry** | `config.py` | The canonical schema and the instrument-type records that drive everything. |

The stages are generic. **Source differences live in the registry as data, not
in code.**

## Adding a new instrument

This is the main extensibility point: **add one record to `REGISTRY` in
`config.py`.** No other file changes.

A record declares how a source's files map onto the canonical schema:

```python
SourceType(
    name="my instrument",       # human name (logs / metadata)
    match="myinst",             # filename substring; discovery matches by this
    is_anchor=False,            # True only for HRRR (defines the snapshots)
    column_map={                # direct canonical -> raw-variable renames
        "x": "lon", "y": "lat", "z": "alt",
        "u": "wind_u", "v": "wind_v",
    },
    derive=my_hook,             # OPTIONAL: computed columns (see below)
    derive_inputs=("base_time", "time_offset"),  # raw vars the hook needs
    chunk_dim="time",           # dimension to stream over (default "time")
)
```

**Direct renames** go in `column_map`. **Computed columns** go in a small
`derive` hook (a pure function of the raw variable dict). Existing hooks handle
ARM absolute time (`base_time + time_offset -> t`) and meteorological wind
(`wspd, wdir -> u, v`).

Any optional column a source omits is left `NaN`, so partial sources are fine
(only `x, y, z, t` are required). A source with **no** canonical field
(e.g. housekeeping) is registered with `canonical=False`: recognized, but
contributes no rows.

Discovery matches files to records by **filename** (ARM datastream naming), so
layout is free and one folder may hold mixed types. Files matching no record are
reported and skipped.

## Usage

```bash
# point at a folder tree of NetCDF files; get split cases + metadata
python orchestrator.py <input_root> <out_dir> \
    [--chunk-size N] [--test-fraction F] [--seed S] [-v | -vv]
```

`<input_root>` is any folder tree of `.nc` / `.cdf` files. Logging is silent by
default; `-v` shows the sync summary, `-vv` per-block detail. Output:

```
<out_dir>/
    metadata.json          # schema, normalization recipe, split, case def
    train/  hrrr_t<epoch>.npy ...
    test/   hrrr_t<epoch>.npy ...
```

### Standalone LES (no HRRR)

To process a single LES case (e.g. FastEddy) without a real HRRR inlet, add a
**dummy anchor**: a minimal HRRR-named `.nc` with a `valid_time` matching the LES
time and `longitude`/`latitude` spanning the LES x/y extent. HRRR contributes no
rows, so the dummy only defines the case window; the LES rows are the data. Use
`--test-fraction 0` when there is just one case.

## Testing

```bash
pytest        # from this directory
```

`fixtures/generate_phony_data.py` writes real-format phony NetCDFs for every
instrument type, including deliberate off-time and off-location rows.
`tests/test_end_to_end.py` runs the whole pipeline and asserts: cases split into
`train/`+`test/` with metadata; every output row is inside the time window and
bbox; off-location/off-time rows are excluded; every type is discovered by
filename; derive hooks are correct; LASSO supplies `theta`/`p_prime` while
sensors keep rows with `NaN`; `canonical=False` types contribute no rows;
normalization is global.

## Requirements

Python with `numpy` and `netCDF4`. (Runs against the project's `ML_venv`.)

## Known limitations / next steps

- **Coordinate reconciliation is not done.** Sources sit in different frames
  (HRRR/sensor degrees, LES metres); spatial matching assumes a shared frame,
  true on the phony data but not yet on the real files.
- **HRRR `theta`/`p'`** are not derived (HRRR stores actual `T`/pressure, not the
  WRF-style perturbations); left `NaN` pending a group decision on the reference
  state.
- **Normalization** is a placeholder per-column max; swap the `Normalizer` body
  once the real scheme is settled.
- **Parallel reads** are designed for (chunks are independent) but not built.
