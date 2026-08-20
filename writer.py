"""Write one ``.npy`` per HRRR snapshot + a shared ``metadata.json``."""

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
            "Coordinate frames not yet reconciled: HRRR/sensor x/y in degrees, "
            "LES in metres, z mixes geopotential height and metres.",
        ],
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
