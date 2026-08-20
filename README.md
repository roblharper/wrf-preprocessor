# WRF-PINN pre-processor

Consolidates heterogeneous atmospheric NetCDF data into **one normalized binary
per HRRR snapshot** that the PINN reads directly. It lives *outside* the model
codebase on purpose: the PINN should never become a data reader. All the messy
differences between sources (variables, units, coordinates, resolutions) are
resolved here, once, so the model always sees exactly one format.

## What it produces

For each HRRR snapshot it writes one `.npy` of rows in a fixed canonical schema:

```
x, y, z, t, u, v, w, T, P
```

plus a shared `metadata.json` (schema, the global normalization recipe, and the
case definition). The PINN's `read_case(name, root)` is then just
"load `root/<name>.npy`".

> The schema is a **working assumption** pending group agreement. Density is not
> stored by any source, so the draft carries temperature and pressure (`T`, `P`)
> and leaves density derivation for later.

## What a "case" is

A case is **one HRRR snapshot** (the inlet the PINN predicts from) bundled with
the LES and sensor data that belong to it. "Belong to it" means, for each
snapshot:

- **time:** within a tolerance window of the snapshot time
  (`TIME_TOLERANCE_SECONDS`, default ±30 min);
- **space:** inside the snapshot's x/y bounding box.

Rows matching no snapshot are **dropped**. Normalization is **global**: one
recipe fit over all synced rows, applied to every case, so they share a scale.

## Pipeline

```
NetCDF sources ─▶ reader ─▶ (derive) ─▶ sync ─▶ normalize ─▶ writer ─▶ <snapshot>.npy
                                                                        + metadata.json
```

| Stage | File | Job |
|-------|------|-----|
| **Reader** | `reader.py` | Stream any NetCDF in chunks (memory-safe); map raw variables to canonical columns by dimension name. |
| **Processor** | `processor.py` | Run a source's `derive` hook (computed columns), assemble canonical rows, then normalize. |
| **Sync** | `sync.py` | Split HRRR into per-snapshot bundles; attach LES/sensor rows by time window + bbox; drop the rest. |
| **Writer** | `writer.py` | One `.npy` per snapshot + shared `metadata.json`. |
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

**Direct renames** go in `column_map`. **Computed columns** (a value that is not
just a rename) go in a small `derive` hook — a pure function of the raw variable
dict. Existing hooks handle:

- ARM absolute time: `base_time + time_offset -> t`
- meteorological wind: `wspd, wdir -> u, v`
- temperature: Celsius `-> T` (Kelvin)

Anything a source does not provide is left as `NaN` in the output (a "not
supplied" marker), so partial sources are fine. A source that carries **no**
canonical field at all (e.g. instrument housekeeping) is registered with
`canonical=False`: it is recognized and documented, ready to carry the day the
schema grows, without being forced into columns that do not fit.

Discovery matches files to records by **filename** (ARM datastream naming), so
files can live in any folder layout and one folder may hold mixed types. Files
matching no record are reported and skipped, never silently forced.

## Usage

```bash
# point at a folder tree of NetCDF files; get per-snapshot .npy + metadata
python orchestrator.py <input_root> <out_dir> [--chunk-size N] [-v | -vv]
```

`<input_root>` is any folder tree of `.nc` / `.cdf` files (in-flight suffixes
like `.cdf.v1` are ignored). Logging is silent by default; `-v` shows the sync
summary and skipped files, `-vv` shows per-block filtering detail. Output:

```
<out_dir>/
    metadata.json              # schema + global normalization recipe + case def
    hrrr_t<epoch>.npy          # one per HRRR snapshot
    ...
```

## Testing

```bash
pytest        # from this directory
```

`fixtures/generate_phony_data.py` writes real-format phony NetCDFs for every
structural instrument type (gridded snapshot, single-point series, vertical
profile, speed/direction wind, Celsius temperature, housekeeping), including
deliberate off-time and off-location rows. `tests/test_end_to_end.py` runs the
whole pipeline and asserts:

- one `.npy` per snapshot + metadata;
- every output row is within the time window **and** bbox (junk dropped);
- the off-location station is excluded; the off-time snapshot has no sync-time data;
- discovery matches every instrument type by filename;
- derive hooks produce correct values (absolute time, u/v from speed/dir, K from C);
- an unmapped (`canonical=False`) type contributes no rows;
- normalization is global.

## Requirements

Python with `numpy` and `netCDF4`. (Runs against the project's `ML_venv`.)

## Known limitations / next steps

- **Coordinate reconciliation is not done.** Sources sit in different frames
  (HRRR/sensor degrees, LES metres, geopotential height vs metres). Cross-source
  spatial matching currently assumes a shared frame — true on the phony test
  data, not yet on the real files. Unifying the frames is the next step.
- **Variable adaptation is partial** (e.g. FastEddy `theta -> T`, deriving
  density) — the hooks exist; the specific conversions are added as needed.
- **Normalization** is a placeholder per-column max (the `Normalizer` class);
  replace its body once the group settles the real scheme.
- **Parallel reads** are designed for (chunks are independent) but not built.
