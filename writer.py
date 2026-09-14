"""Write per-case ``.npy`` into train/ and test/ + a metadata.json.

Also holds the mergeable-stats seam: each case drops a tiny stats sidecar in
phase 1; ``consolidate_stats`` merges them into one global min/max for the recipe.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config import CANONICAL_COLUMNS, SRC_SIM, SRC_SENSOR


def write_case_stats(work: Path, case_id: str, col_min: np.ndarray,
                     col_max: np.ndarray) -> None:
    """Write a case's per-column min/max sidecar (NaN -> null) beside its raw .npy."""
    (work / f"{case_id}.stats.json").write_text(json.dumps({
        "columns": list(CANONICAL_COLUMNS),
        "min": _jsonable(col_min),
        "max": _jsonable(col_max),
    }))


def consolidate_stats(work: Path, case_ids) -> tuple[np.ndarray, np.ndarray]:
    """Merge every case's stats sidecar into one global (min, max) per column.

    NaN entries (a column no case in this id ever supplied) stay NaN; the
    Normalizer maps such columns to the identity.
    """
    ncol = len(CANONICAL_COLUMNS)
    gmin = np.full(ncol, np.nan)
    gmax = np.full(ncol, np.nan)
    for cid in case_ids:
        s = json.loads((work / f"{cid}.stats.json").read_text())
        cmin = _from_json(s["min"])
        cmax = _from_json(s["max"])
        gmin = np.fmin(gmin, cmin)   # fmin/fmax ignore NaN, so first real value wins
        gmax = np.fmax(gmax, cmax)
    return gmin, gmax


def _jsonable(arr: np.ndarray) -> list:
    return [None if not np.isfinite(v) else float(v) for v in arr]


def _from_json(vals: list) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in vals], dtype=np.float64)


class CaseWriter:
    """Stream cases straight into train/ or test/ (no staging copy).

    The train/test split is decided up front from the known case ids, so each
    normalized case is written once to its final home. Peak RAM = one case.
    """

    def __init__(self, out_dir: Path, normalization_recipe: dict, case_ids, *,
                 test_fraction: float = 0.2, seed: int = 0,
                 hrrr_times: dict[str, float] | None = None) -> None:
        self._out = out_dir
        self._recipe = normalization_recipe
        self._test_fraction = test_fraction
        self._seed = seed
        self._hrrr_times = hrrr_times or {}
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
                        self._test_fraction, self._seed, self._hrrr_times)
        return self._written


def _split_ids(ids: list[str], test_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    """Shuffle case ids by seed and split off ``test_fraction`` for testing."""
    order = np.random.default_rng(seed).permutation(len(ids))
    # don't generate test data if test_fraction = 0
    if test_fraction <= 0:
        n_test = 0
    else:
        n_test = max(1, round(len(ids) * test_fraction))
    test = {ids[i] for i in order[:n_test]}
    return [i for i in ids if i not in test], [i for i in ids if i in test]


def _write_metadata(
    out_dir: Path,
    normalization_recipe: dict,
    counts: dict[str, int],
    written: dict[str, list[Path]],
    test_fraction: float,
    seed: int,
    hrrr_times: dict[str, float],
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
            "unit": "one input folder: an HRRR snapshot condition plus all LES/sensor rows placed in it",
            "case_id": "the folder name",
            "hrrr_role": "tags the case with its snapshot time; contributes no rows",
        },
        "split": {"test_fraction": test_fraction, "seed": seed},
        "normalization": normalization_recipe,
        "cases": {
            group: {
                p.stem: {
                    "rows": counts[p.stem],
                    "hrrr_time": hrrr_times.get(p.stem),
                }
                for p in paths
            }
            for group, paths in written.items()
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
