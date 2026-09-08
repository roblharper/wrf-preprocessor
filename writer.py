"""Write per-snapshot ``.npy`` cases into train/ and test/ + a metadata.json."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config import CANONICAL_COLUMNS, TIME_TOLERANCE_SECONDS, SRC_SIM, SRC_SENSOR


class CaseWriter:
    """Stream cases straight into train/ or test/ (no staging copy).

    The train/test split is decided up front from the known case ids, so each
    normalized case is written once to its final home. Peak RAM = one case,
    peak extra disk = one case.
    """

    def __init__(self, out_dir: Path, normalization_recipe: dict, case_ids, *,
                 test_fraction: float = 0.2, seed: int = 0) -> None:
        self._out = out_dir
        self._recipe = normalization_recipe
        self._test_fraction = test_fraction
        self._seed = seed
        train_ids, test_ids = _split_ids(sorted(case_ids), test_fraction, seed)
        self._group = {cid: "train" for cid in train_ids}
        self._group.update({cid: "test" for cid in test_ids})
        for name in ("train", "test"):
            (out_dir / name).mkdir(parents=True, exist_ok=True)
        self._counts: dict[str, int] = {}
        self._written: dict[str, list[Path]] = {"train": [], "test": []}

    def add(self, case_id: str, data: np.ndarray) -> None:
        if not data.size:
            return
        group = self._group[case_id]
        path = self._out / group / f"{case_id}.npy"
        np.save(path, data)
        self._counts[case_id] = int(data.shape[0])
        self._written[group].append(path)

    def finalize(self) -> dict[str, list[Path]]:
        _write_metadata(self._out, self._recipe, self._counts, self._written,
                        self._test_fraction, self._seed)
        return self._written


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

    ids = [cid for cid, data in cases.items() if data.size]
    writer = CaseWriter(out_dir, normalization_recipe, ids,
                        test_fraction=test_fraction, seed=seed)
    for cid, data in cases.items():
        writer.add(cid, data)
    return writer.finalize()


def _split_ids(ids: list[str], test_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    """Shuffle case ids by seed and split off ``test_fraction`` for testing."""
    order = np.random.default_rng(seed).permutation(len(ids))
    n_test = max(1, round(len(ids) * test_fraction)) if ids else 0
    test = {ids[i] for i in order[:n_test]}
    return [i for i in ids if i not in test], [i for i in ids if i in test]


def _write_metadata(
    out_dir: Path,
    normalization_recipe: dict,
    counts: dict[str, int],
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
            "source_codes": {SRC_SIM: "simulation", SRC_SENSOR: "sensor"},
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
            group: {p.stem: counts[p.stem] for p in paths}
            for group, paths in written.items()
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
