"""Output stage: one ``.npy`` per HRRR snapshot + a shared metadata sidecar.

Layout (matches the ``Case`` interface the PINN expects):

    <out_dir>/
        metadata.json          schema + global normalization recipe (shared)
        <snapshot_id>.npy      one file per HRRR snapshot; canonical column order
        ...

so the PINN's ``read_case(name, root)`` is just "load ``root/<name>.npy``".

The per-case grouping is decided upstream by the syncing stage (one bundle per
HRRR snapshot). This writer just serializes those bundles. Normalization is
global: one recipe, recorded once, already applied to every bundle.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config import CANONICAL_COLUMNS, TIME_TOLERANCE_SECONDS


def write_cases(
    out_dir: Path,
    cases: dict[str, np.ndarray],
    normalization_recipe: dict,
) -> list[Path]:
    """Write each snapshot bundle to ``<id>.npy`` and one shared metadata.json."""

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for case_id, data in sorted(cases.items()):
        if data.size == 0:
            continue
        path = out_dir / f"{case_id}.npy"
        np.save(path, data)
        written.append(path)

    _write_metadata(out_dir, normalization_recipe, cases, written)
    return written


def _write_metadata(
    out_dir: Path,
    normalization_recipe: dict,
    cases: dict[str, np.ndarray],
    written: list[Path],
) -> None:
    metadata = {
        "schema": {
            "columns": list(CANONICAL_COLUMNS),
            "dtype": "float64",
            "layout": "rows x columns, columns in schema order",
            "missing_value": "NaN (column not supplied by source)",
        },
        "case_definition": {
            "unit": "one HRRR snapshot plus co-located, co-temporal LES/sensor rows",
            "time_tolerance_seconds": TIME_TOLERANCE_SECONDS,
            "space_match": "inside the HRRR snapshot's x/y bounding box",
            "unmatched_rows": "dropped",
        },
        "normalization": normalization_recipe,
        "cases": {p.stem: int(cases[p.stem].shape[0]) for p in written},
        "known_limitations": [
            "Coordinate frames are NOT yet reconciled across sources: HRRR and "
            "sensor x/y are lon/lat in degrees, simulation x/y are metres, z mixes "
            "geopotential height and metres. Cross-source rows in one snapshot are "
            "matched by raw x/y until the reconciliation step is implemented.",
        ],
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
