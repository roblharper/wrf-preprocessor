"""Write per-snapshot ``.npy`` cases into train/ and test/ + a metadata.json."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config import CANONICAL_COLUMNS, TIME_TOLERANCE_SECONDS, SRC_INLET, SRC_SIM, SRC_SENSOR


def write_cases(
    out_dir: Path,
    cases: dict[str, np.ndarray],
    normalization_recipe: dict,
    *,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> dict[str, list[Path]]:
    """Split cases (by whole case) into ``train/`` and ``test/`` and write them.

    Returns ``{"train": [...], "test": [...]}``. The split is seeded and
    reproducible; normalization is global (already applied), recorded once in the
    shared metadata.json.
    """

    non_empty = sorted(cid for cid, data in cases.items() if data.size)
    train_ids, test_ids = _split_ids(non_empty, test_fraction, seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    written = {
        "train": _write_group(out_dir / "train", train_ids, cases),
        "test": _write_group(out_dir / "test", test_ids, cases),
    }
    _write_metadata(out_dir, normalization_recipe, cases, written, test_fraction, seed)
    return written


def _split_ids(ids: list[str], test_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    """Shuffle case ids by seed and split off ``test_fraction`` for testing."""
    order = np.random.default_rng(seed).permutation(len(ids))
    n_test = max(1, round(len(ids) * test_fraction)) if ids else 0
    test = {ids[i] for i in order[:n_test]}
    return [i for i in ids if i not in test], [i for i in ids if i in test]


def _write_group(group_dir: Path, ids: list[str], cases: dict[str, np.ndarray]) -> list[Path]:
    group_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for case_id in ids:
        path = group_dir / f"{case_id}.npy"
        np.save(path, cases[case_id])
        paths.append(path)
    return paths


def _write_metadata(
    out_dir: Path,
    normalization_recipe: dict,
    cases: dict[str, np.ndarray],
    written: dict[str, list[Path]],
    test_fraction: float,
    seed: int,
) -> None:
    metadata = {
        "schema": {
            "columns": list(CANONICAL_COLUMNS),
            "dtype": "float64",
            "layout": "rows x columns, columns in schema order",
            "missing_value": "NaN (column not supplied by source)",
            "source_codes": {SRC_INLET: "inlet", SRC_SIM: "simulation",
                             SRC_SENSOR: "sensor"},
        },
        "case_definition": {
            "unit": "one HRRR snapshot plus co-located, co-temporal LES/sensor rows",
            "time_tolerance_seconds": TIME_TOLERANCE_SECONDS,
            "space_match": "inside the HRRR snapshot's x/y bounding box",
            "unmatched_rows": "dropped",
        },
        "split": {"test_fraction": test_fraction, "seed": seed},
        "normalization": normalization_recipe,
        "cases": {
            group: {p.stem: int(cases[p.stem].shape[0]) for p in paths}
            for group, paths in written.items()
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
