"""Test the conditional case generator: raw sub-domain .npz -> normalized cases.

Checks global min/max normalization lands state vars in [0, 1] consistently across
members (phi, psi, targets), the train/test split, and that the written blobs carry
every member the pinn's reader expects.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import conditional_cases as cc  # noqa: E402


def _make_raw(path: Path, seed: int) -> None:
    """Write one raw sub-domain .npz with known small shapes."""
    nv, nc, nf = len(cc.STATE_VARS), len(cc.COORD_NAMES), len(cc.FACE_NAMES)
    n_ic, n_times, face_len, n_terr, n_pts = 5, 3, 4, 6, 7
    rng = np.random.default_rng(seed)
    np.savez(
        path,
        initial=rng.standard_normal((n_ic, nc + nv)).astype(np.float32) * 5,
        boundary=rng.standard_normal((nf, n_times, face_len, nv)).astype(np.float32) * 5,
        terrain=rng.standard_normal((n_terr, 3)).astype(np.float32),
        interior=rng.standard_normal((n_pts, nc)).astype(np.float32) * 5,
        targets=rng.standard_normal((n_pts, nv)).astype(np.float32) * 5,
        times=np.arange(n_times, dtype=np.float32),
    )


def test_generate_normalizes_and_splits(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir()
    for i in range(10):
        _make_raw(raw / f"sub_{i:02d}.npz", seed=i)

    out = tmp_path / "out"
    written = cc.generate(raw, out, test_fraction=0.2, seed=0)

    assert len(written["train"]) == 8 and len(written["test"]) == 2

    # every normalized state value lands in [0, 1] (global minmax_01)
    nc = len(cc.COORD_NAMES)
    for group in ("train", "test"):
        for p in written[group]:
            with np.load(p) as b:
                for member, sl in (("initial", np.s_[:, nc:]),
                                   ("targets", np.s_[:]),
                                   ("boundary", np.s_[:])):
                    vals = b[member][sl]
                    vals = vals[np.isfinite(vals)]
                    assert vals.min() >= -1e-5 and vals.max() <= 1 + 1e-5, member

    meta = json.loads((out / "metadata.json").read_text())
    assert meta["normalization"]["method"] == "minmax_01"
    assert meta["schema"]["state_vars"] == list(cc.STATE_VARS)


def test_written_cases_load_in_pinn(tmp_path):
    """The preprocessor output must load via the pinn's reader (contract closes)."""
    pytest.importorskip("wrf_pinn")
    from wrf_pinn.data.conditional_case import read_conditional_case

    raw = tmp_path / "raw"; raw.mkdir()
    for i in range(4):
        _make_raw(raw / f"sub_{i:02d}.npz", seed=i)
    out = tmp_path / "out"
    written = cc.generate(raw, out, test_fraction=0.25, seed=0)

    case = read_conditional_case(written["train"][0])
    assert case.boundary.shape[0] == len(cc.FACE_NAMES)
    assert case.targets.shape[1] == len(cc.STATE_VARS)
    assert np.isfinite(case.interior).all()
