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
    """Write one raw sub-domain .npz with known small shapes.

    ``initial`` must be a structured nx*ny*nz grid (the encoder reshapes it over
    z-levels), so its coordinates are a real meshgrid, not random points.
    """
    nv, nc, nf = len(cc.STATE_VARS), len(cc.COORD_NAMES), len(cc.FACE_NAMES)
    ngrid = (3, 3, 4)                       # nx, ny, nz
    n_ic = ngrid[0] * ngrid[1] * ngrid[2]
    n_times, face_len, n_terr, n_pts = 3, 4, 6, 7
    rng = np.random.default_rng(seed)
    gx, gy, gz = (a.reshape(-1) for a in np.meshgrid(
        np.arange(ngrid[0]), np.arange(ngrid[1]), np.arange(ngrid[2]), indexing="ij"))
    ini_coords = np.stack([gx, gy, gz, np.full(n_ic, seed)], 1).astype(np.float32)
    ini_state = rng.standard_normal((n_ic, nv)).astype(np.float32) * 5
    n_face_pts = 9
    surface = rng.standard_normal((n_times, n_face_pts, 5)).astype(np.float32) * 5
    np.savez(
        path,
        initial=np.concatenate([ini_coords, ini_state], axis=1),
        boundary=rng.standard_normal((nf, n_times, face_len, nv)).astype(np.float32) * 5,
        boundary_coords=rng.standard_normal((nf, n_times, face_len, nc)).astype(np.float32) * 5,
        terrain=rng.standard_normal((n_terr, 3)).astype(np.float32),
        interior=rng.standard_normal((n_pts, nc)).astype(np.float32) * 5,
        targets=rng.standard_normal((n_pts, nv)).astype(np.float32) * 5,
        times=np.arange(n_times, dtype=np.float32),
        surface=surface,
    )


def test_generate_normalizes_and_splits(tmp_path):
    pytest.importorskip("wrf_pinn")
    raw = tmp_path / "raw"; raw.mkdir()
    for i in range(10):
        _make_raw(raw / f"sub_{i:02d}.npz", seed=i)

    out = tmp_path / "out"
    written = cc.generate(raw, out, test_fraction=0.2, seed=0)

    assert len(written["train"]) == 8 and len(written["test"]) == 2

    # Normalized targets land in [0, 1]; the POD-decoded initial/boundary fields
    # reconstruct back into [0, 1] (coeffs/modes themselves are unbounded).
    from wrf_pinn.data.conditional_case import decode_boundary, decode_initial
    for group in ("train", "test"):
        for p in written[group]:
            with np.load(p) as b:
                tv = b["targets"]; tv = tv[np.isfinite(tv)]
                assert tv.min() >= -1e-5 and tv.max() <= 1 + 1e-5, "targets"
                for name, decoded in (("boundary", decode_boundary(b)),
                                      ("initial", decode_initial(b))):
                    for var, field in decoded.items():
                        v = field[np.isfinite(field)]
                        assert v.min() >= -1e-2 and v.max() <= 1 + 1e-2, f"{name}:{var}"

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
    # decoded boundary is (n_state, n_faces, n_times, face_len)
    assert case.boundary.shape[0] == len(cc.STATE_VARS)
    assert case.boundary.shape[1] == len(cc.FACE_NAMES)
    assert case.targets.shape[1] == len(cc.STATE_VARS)
    assert np.isfinite(case.interior).all()


def _shared(tmp_path, n=10, k=3, workers=1, name="out"):
    raw = tmp_path / "raw"
    if not raw.exists():
        raw.mkdir()
        for i in range(n):
            _make_raw(raw / f"sub_{i:02d}.npz", seed=i)
    out = tmp_path / name
    written = cc.generate(raw, out, test_fraction=0.2, seed=0, workers=workers,
                          ks_initial=cc._default_ks(k), initial_basis="shared")
    return raw, out, written


def test_shared_basis_is_one_basis_fit_on_train_only(tmp_path):
    """Every case carries the same modes; they match an SVD of the train cases'
    level-mean-removed initial fields; test cases do not enter the fit."""
    raw, out, written = _shared(tmp_path)
    phi = np.load(out / "shared_basis.npz")
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["initial_basis"] == {"kind": "shared", "fit_cases": 8,
                                     "file": "shared_basis.npz"}
    train = {p.stem for p in written["train"]}
    for var_j, var in enumerate(cc.STATE_VARS):
        rows = []
        for p in sorted(raw.glob("*.npz")):
            if p.stem in train:
                g = cc._initial_grids(np.load(p)["initial"])[1][var_j].astype(np.float64)
                rows.append(g - g.mean(axis=1, keepdims=True))
        Vt = np.linalg.svd(np.concatenate(rows), full_matrices=False)[2][:3]
        # same subspace (sign/rotation free): projector difference ~ 0
        P_ref, P = Vt.T @ Vt, phi[f"modes_{var}"] @ phi[f"modes_{var}"].T
        assert np.abs(P_ref - P).max() < 1e-4, var
        for g in ("train", "test"):
            for p in written[g]:
                with np.load(p) as b:
                    np.testing.assert_array_equal(b[f"ini_modes_{var}"][:, 1:],
                                                  phi[f"modes_{var}"])


def test_shared_basis_coeffs_mean_where(tmp_path):
    """Unlike per-case POD, shuffling a tile's columns changes its coefficients,
    and decode(encode(x)) is x projected onto [1 | Phi] per level."""
    _, out, written = _shared(tmp_path)
    phi = np.load(out / "shared_basis.npz")["modes_u"].astype(np.float64)
    case = cc._normalize_case(cc._load_raw(tmp_path / "raw" / f"{written['test'][0].stem}.npz"),
                              json.loads((out / "metadata.json").read_text())["normalization"])
    grid = cc._initial_grids(case["initial"])[1][0]
    _, modes, coeffs = cc._shared_encode(grid, phi.astype(np.float32))
    m = grid.mean(axis=1, keepdims=True)
    np.testing.assert_allclose(coeffs @ modes.T, m + ((grid - m) @ phi) @ phi.T, atol=1e-5)
    perm = np.random.default_rng(0).permutation(grid.shape[1])
    shuffled = cc._shared_encode(grid[:, perm], phi.astype(np.float32))[2]
    assert not np.allclose(shuffled[:, 1:], coeffs[:, 1:], atol=1e-3)
    np.testing.assert_allclose(shuffled[:, 0], coeffs[:, 0], atol=1e-6)   # level means


def test_shared_basis_parallel_matches_serial(tmp_path):
    """Gram sums merge across workers; the written cases are identical."""
    _, serial, _ = _shared(tmp_path, name="serial")
    _, pooled, _ = _shared(tmp_path, workers=2, name="pooled")
    a, b = np.load(serial / "shared_basis.npz"), np.load(pooled / "shared_basis.npz")
    for k in a.files:
        np.testing.assert_allclose(a[k], b[k], atol=1e-6)


def test_shared_basis_loads_in_pinn(tmp_path):
    """wrf_pinn reads shared-basis cases unchanged; z holds [m | a] per level."""
    pytest.importorskip("wrf_pinn")
    from wrf_pinn.data.conditional_case import read_conditional_case
    _, _, written = _shared(tmp_path, k=3)
    with np.load(written["train"][0]) as b:
        assert b["ini_coeffs_u"].shape == (4, 4)                 # nz, 1 + k
    case = read_conditional_case(written["train"][0])
    assert np.isfinite(case.z).all() and np.isfinite(case.initial).all()


def test_shared_basis_rejects_k_too_large(tmp_path):
    with pytest.raises(ValueError, match="k_initial"):
        _shared(tmp_path, k=9)                                   # grid has 9 columns


def test_keep_raw_stores_normalized_fields(tmp_path):
    """--keep-raw writes the initial and boundary fields themselves, normalized by
    the same recipe as targets, alongside the unchanged POD pieces."""
    raw = tmp_path / "raw"; raw.mkdir()
    for i in range(5):
        _make_raw(raw / f"sub_{i:02d}.npz", seed=i)
    pod = cc.generate(raw, tmp_path / "pod", test_fraction=0.2, seed=0)
    kept = cc.generate(raw, tmp_path / "kept", test_fraction=0.2, seed=0, keep_raw=True)
    recipe = json.loads((tmp_path / "kept" / "metadata.json").read_text())["normalization"]
    off, scale = np.array(recipe["state_offset"]), np.array(recipe["state_scale"])
    for p_pod, p_kept in zip(pod["train"] + pod["test"], kept["train"] + kept["test"]):
        src = np.load(raw / p_kept.name)
        with np.load(p_pod) as a, np.load(p_kept) as b:
            for k in a.files:
                np.testing.assert_array_equal(a[k], b[k], err_msg=k)
            want_bnd = (src["boundary"] - off) / scale
            np.testing.assert_allclose(b["bnd_raw"], want_bnd, rtol=1e-5, atol=1e-6)
            grids = cc._initial_grids(src["initial"])[1]           # (n_state, nz, nxy)
            want_ini = (grids - off[:, None, None]) / scale[:, None, None]
            np.testing.assert_allclose(b["ini_raw"], want_ini, rtol=1e-5, atol=1e-6)


def test_recipe_from_reuses_normalization_and_split(tmp_path):
    """A subset re-run with recipe_from keeps the full run's recipe and puts each
    case in the same group, so test scores stay comparable."""
    raw = tmp_path / "raw"; raw.mkdir()
    for i in range(10):
        _make_raw(raw / f"sub_{i:02d}.npz", seed=i)
    full = cc.generate(raw, tmp_path / "full", test_fraction=0.3, seed=1)
    sub = tmp_path / "sub"; sub.mkdir()
    for i in (1, 4, 6, 9):
        (sub / f"sub_{i:02d}.npz").write_bytes((raw / f"sub_{i:02d}.npz").read_bytes())
    meta_full = tmp_path / "full" / "metadata.json"
    part = cc.generate(sub, tmp_path / "part", recipe_from=meta_full, keep_raw=True)
    full_meta = json.loads(meta_full.read_text())
    part_meta = json.loads((tmp_path / "part" / "metadata.json").read_text())
    assert part_meta["normalization"] == full_meta["normalization"]
    for g in ("train", "test"):
        assert {p.stem for p in part[g]} == {p.stem for p in full[g]} & {p.stem for p in part["train"] + part["test"]}
        for p in part[g]:
            with np.load(p) as a, np.load(tmp_path / "full" / g / p.name) as b:
                np.testing.assert_array_equal(a["targets"], b["targets"])


def test_recipe_from_rejects_unknown_cases(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir()
    for i in range(3):
        _make_raw(raw / f"sub_{i:02d}.npz", seed=i)
    cc.generate(raw, tmp_path / "full", test_fraction=0.3, seed=0)
    _make_raw(raw / "extra.npz", seed=7)
    with pytest.raises(ValueError, match="not in"):
        cc.generate(raw, tmp_path / "again", recipe_from=tmp_path / "full" / "metadata.json")
