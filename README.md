# WRF-PINN pre-processor

Consolidates heterogeneous atmospheric NetCDF data into **one normalized binary
per case**, ready to load directly. All the differences between sources
(variables, units, coordinates, resolutions) are resolved here, once, so a
consumer sees exactly one format.

## What it produces

For each case it writes one `.npy` of rows in a fixed canonical schema:

```
x, y, z, t, u, v, w, theta, p_prime, source
```

plus a shared `metadata.json` (schema, the global normalization recipe, the
source-code map, the train/test split, and the case definition). The cases are
written into `train/` and `test/` subfolders (see below).

`x, y, z, t` are **required**; a row missing any of them is dropped. The physical
columns `u, v, w, theta, p_prime` are **optional**: a source that does not
measure one leaves it `NaN` (the row is kept and the gap is logged). `source` is
a per-row integer tag identifying the data category the row came from, so a
consumer can weight it per source.

## What a "case" is

**A case is one input folder.** You lay out the input as
`input_root/<case_name>/`, and each folder holds one HRRR file plus all the LES
and sensor files you want in that case. The **folder name is the case id** (it
must be unique). The tool assumes nothing about which data belongs where — the
directory layout *declares* it. You already know the data↔snapshot association
(you built the LES run for a specific HRRR inlet), so it is stated, not inferred.

HRRR is **not** a data source: it only **tags the case** with its snapshot time
(recorded in `metadata.json` for traceability) and contributes no rows. Every
non-HRRR file in a folder is data **unconditionally** — there is no time window,
no bounding box, no matching. All rows from every file in the folder are
concatenated into that case's table.

Normalization is **global** and **affine**: one recipe
(`value = offset + scale * normalized`) fit over all cases' rows and applied to
every case, so they share a scale. The scheme is swappable (see `NORMALIZERS` in
`processor.py`); the default (`minmax_01`) maps each column to `[0, 1]`.

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

Two phases with a **mergeable-stats seam** between them, so the expensive
per-case work is independent and resumable:

```
phase 1 (per case): read files ─▶ (derive) ─▶ canonical ─▶ concat ─▶ raw <case>.npy
                                                                    + <case>.stats.json
consolidate:        merge every <case>.stats.json ─▶ one global min/max ─▶ recipe
phase 2 (per case): re-read raw <case>.npy ─▶ normalize ─▶ writer ─▶ <case>.npy
                                                                    + metadata.json
```

| Stage | File | Job |
|-------|------|-----|
| **Reader** | `reader.py` | `discover_cases` (one Case per subfolder); stream any NetCDF in chunks (memory-safe); map raw variables to canonical columns by dimension name. |
| **Processor** | `processor.py` | Run a source's `derive` hook (computed columns), assemble canonical rows; streaming per-column min/max; affine normalize. |
| **Sync** | `sync.py` | Read an HRRR file's snapshot time to tag its case (no grouping). |
| **Writer** | `writer.py` | Per-case stats sidecar + `consolidate_stats`; split cases into `train/`/`test/`; write each `.npy` + a shared `metadata.json`. |
| **Orchestrator** | `orchestrator.py` | Wire the two phases + consolidation + CLI. |
| **Device** | `device.py` | One place that swaps the array backend: numpy on CPU, torch on GPU. Only the swap points live here (no abstraction layer). |
| **Registry** | `config.py` | The canonical schema and the instrument-type records that drive everything. |

The stages are generic. **Source differences live in the registry as data, not
in code.** The same generic property holds for the compute backend: every stage's
array math is written once and runs on numpy or torch depending on `--device`, so
CPU and GPU share one code path (see **GPU compute** below).

## Adding a new instrument

This is the main extensibility point: **add one record to `REGISTRY` in
`config.py`.** No other file changes.

A record declares how a source's files map onto the canonical schema:

```python
SourceType(
    name="my instrument",       # human name (logs / metadata)
    match="myinst",             # filename substring; discovery matches by this
    is_anchor=False,            # True only for HRRR (tags the case time)
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

Within a case folder, files are matched to records by **filename** (ARM
datastream naming) only to pick the reader for each file, not to group them.
Files matching no record are reported and skipped.

## Usage

```bash
# point at a folder of case subfolders; get split cases + metadata
python orchestrator.py <input_root> <out_dir> \
    [--chunk-size N] [--test-fraction F] [--seed S] [--device cpu|cuda] [-v | -vv]
```

`<input_root>` holds one subfolder per case, each with one HRRR file plus its
LES/sensor `.nc` / `.cdf` files:

```
<input_root>/
    <case_a>/  hrrr.nc  wrfout_d01.nc  sgpecorsfwind...nc  ...
    <case_b>/  hrrr.nc  wrfout_d01.nc  ...
```

Logging is silent by default; `-v` shows per-case row counts, `-vv` per-block
detail. Output (case files named by folder):

```
<out_dir>/
    metadata.json          # schema, normalization recipe, split, per-case HRRR time
    train/  <case_a>.npy ...
    test/   <case_b>.npy ...
```

## GPU compute

The array math (destagger, broadcast, canonical assembly, min/max stats,
normalize) is the same code on CPU and GPU; `--device` picks the backend.
`device.py` holds the only backend-aware lines — everything else takes an array
module (`xp`, numpy or torch) and does not know or care which it got. This is the
same "differences are data, not code" principle the source registry follows,
applied to the compute backend.

Cases are **independent**: each is read and assembled on its own, so the
parallelism is at the case level and peak memory is one case's rows regardless of
backend. `--device` only chooses where a case's block math runs.

**CPU (`--device cpu`, default).** numpy throughout; no GPU needed. Use on a
workstation or when a dataset is far larger than device memory.

**GPU (`--device cuda`).** Phase 1 runs a case's read/assemble/min-max on the
GPU; the raw case is copied to host once and spilled as a raw `.npy`. Phase 2
re-reads the raw `.npy`, normalizes it **on the GPU** (offset/scale resident),
and copies it back once to write. The global min/max needed before any normalize
comes from merging the tiny per-case stats sidecars, so there is no all-cases
resident batch to size.

**Where the per-case time goes.** The normalize step is elementwise math over a
whole case; on the CPU it cost ~1.4 s/case. Running it **on the GPU before the
device→host copy** drops it to ~0.007 s/case — effectively free. What remains is
I/O: the netCDF read, the device↔host copies, and the `.npy` read/write. Those
stay on the CPU by design and are shared by both paths — the GPU does not help
with I/O.

> **On benchmarking honestly:** the CPU reference (pre-process ~17.7 s/case,
> measured on **MCC** over 600 LASSO cases) and the GPU runs (**ECC** H100) use
> the **same-size data** — the ECC cases are full 1.2 GB, 144×144×226 wrfout files
> (6 snapshots each), identical in grid and `.npy` size to the real LASSO
> snapshots. The remaining difference is **hardware** (MCC vs ECC), so treat any
> ratio as indicative, not exact; a fully clean figure needs both paths on the
> same machine. What is firmly established: the normalize compute cost is
> eliminated, and both stages are otherwise I/O-bound.

> The GPU path is backend-generic: it operates on the canonical schema, never on
> a source identity, so it applies equally to FastEddy, ARM sensors, or LASSO.

### Standalone LES (no HRRR)

A case folder does not strictly need an HRRR file — without one the case simply
carries no `hrrr_time` tag in metadata. If you want the snapshot time recorded
(e.g. for a FastEddy case), drop in a minimal HRRR-named `.nc` with a matching
`valid_time`; it contributes no rows either way. Use `--test-fraction 0` when
there is just one case.

## Testing

```bash
pytest        # from this directory
```

`fixtures/generate_phony_data.py` writes a folder-per-case tree: each case folder
holds real-format phony NetCDFs for every instrument type plus an HRRR tag.
`tests/test_end_to_end.py` runs the whole pipeline and asserts: case ids come
from folder names; every file in a folder contributes; duplicate folder ids
error; the per-case stats consolidate to a correct global recipe; cases split
into `train/`+`test/` with the recipe and per-case HRRR time in metadata; every
type is discovered by filename; derive hooks are correct; LASSO supplies
`theta`/`p_prime` while sensors keep rows with `NaN`; `canonical=False` types
contribute no rows; output is normalized (training-ready).

## Requirements

Python with `numpy` and `netCDF4`.

## Known limitations / next steps

- **Coordinate reconciliation is not done.** Sources sit in different frames
  (HRRR/sensor degrees, LES metres); the tool concatenates a folder's rows as-is,
  so a consumer that needs a shared frame must reconcile them.
- **HRRR `theta`/`p'`** are moot for rows (HRRR contributes none); it only tags
  the case time.
- **Normalization** defaults to affine min-max into `[0, 1]` (`minmax_01`); add or
  select another scheme in `NORMALIZERS` (`processor.py`).
- **GPU compute** runs the array math on `--device cuda` (see **GPU compute**);
  reads and writes stay on the CPU (I/O-bound). Cutting the case dtype to
  float32 would roughly halve the remaining host-copy + write cost — not yet done.
