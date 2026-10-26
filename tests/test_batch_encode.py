"""batch_encode (torch, batched) must match conditional_cases' per-case numpy path.

SVD signs can flip between backends, so POD pieces are compared through the
truncated reconstruction mean + coeffs @ modes.T, and modes up to sign.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch = pytest.importorskip("torch")
import batch_encode as be  # noqa: E402
import conditional_cases as cc  # noqa: E402

NX, NY, NZ, NT, B = 8, 6, 12, 5, 4
NV, NCO = len(cc.STATE_VARS), len(cc.COORD_NAMES)


def _raw_batch(seed: int = 0) -> dict:
    """B same-shape sub-domains laid out like the carve's raw .npz, stacked."""
    rng = np.random.default_rng(seed)
    gx, gy, gz = (a.reshape(-1) for a in np.meshgrid(
        np.arange(NX), np.arange(NY), np.arange(NZ), indexing="ij"))
    n_ic, fl, n_pts = NX * NY * NZ, NZ * NX, (NT - 1) * NX * NY * NZ
    coords = np.stack([gx, gy, gz * 10.0, np.zeros(n_ic)], 1)
    # smooth-ish fields with a dominant mode so singular values are well separated
    state = (rng.standard_normal((B, n_ic, NV)) * [3, 2, 0.5, 1, 50, 2e-3, 0.3]
             + np.sin(gz / 3.0)[None, :, None] * 10 + [5, -2, 0, 300, 1e5, 8e-3, 0.5])
    targets = rng.standard_normal((B, n_pts, NV)) * 4
    targets[0, :10, 2] = np.nan
    f32 = lambda a: np.asarray(a, dtype=np.float32)
    return {
        "initial": f32(np.concatenate([np.broadcast_to(coords, (B, n_ic, NCO)), state], -1)),
        "boundary": f32(rng.standard_normal((B, 4, NT, fl, NV)) * 4
                        + np.linspace(0, 3, NT)[None, None, :, None, None]),
        "boundary_coords": f32(rng.standard_normal((B, 4, NT, fl, NCO)) * 50),
        "terrain": f32(rng.standard_normal((B, NX * NY, 3))),
        "interior": f32(rng.standard_normal((B, n_pts, NCO)) * 50),
        "targets": f32(targets),
        "times": f32(np.broadcast_to(np.arange(NT) * 60.0, (B, NT))),
        "surface": f32(rng.standard_normal((B, NT, NX * NY, 5)) * 20),
    }


def _recipe(raw: dict) -> dict:
    stats = cc._MinMax.empty()
    for b in range(B):
        stats.update_coord(raw["initial"][b, :, :NCO])
        stats.update_state(raw["initial"][b, :, NCO:])
        stats.update_state(raw["boundary"][b])
        stats.update_coord(raw["interior"][b])
    return cc._norm_recipe(stats)


@pytest.mark.parametrize("keep_raw", [False, True])
def test_matches_numpy_path(keep_raw):
    raw = _raw_batch()
    recipe = _recipe(raw)
    ks_i = {v: 6 for v in cc.STATE_VARS}
    ks_b = {v: 3 for v in cc.STATE_VARS}
    got = be.encode_batch(raw, recipe, ks_i, ks_b, (NX, NY, NZ), keep_raw)
    assert len(got) == B
    for b in range(B):
        ref = cc._encode_case(cc._normalize_case({k: v[b] for k, v in raw.items()}, recipe),
                              ks_i, ks_b, None, keep_raw)
        assert set(got[b]) == set(ref)
        for key, want in ref.items():
            have = got[b][key]
            assert have.shape == want.shape and have.dtype == want.dtype, key
            if "_modes_" in key:                      # sign-free
                np.testing.assert_allclose(np.abs(have), np.abs(want), atol=2e-3, err_msg=key)
            elif "_coeffs_" not in key:
                np.testing.assert_allclose(have, want, rtol=1e-5, atol=1e-5, err_msg=key)
        for pre, names in (("ini", cc.STATE_VARS),
                           ("bnd", [f"{v}_{f}" for v in cc.STATE_VARS for f in range(4)])):
            for n in names:
                rec = lambda c: c[f"{pre}_mean_{n}"] + c[f"{pre}_coeffs_{n}"] @ c[f"{pre}_modes_{n}"].T
                np.testing.assert_allclose(rec(got[b]), rec(ref), atol=1e-4,
                                           err_msg=f"{pre} {n} case {b}")
