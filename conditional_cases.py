"""Normalize raw sub-domain .npz files into ConditionalCase blobs the PINN reads.

Two passes: merge one global min/max recipe over all sub-domains, then normalize
each. Reads .npz only (no netCDF); source agnostic.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

#: Variable order carried in the state columns of every member. Matches the
#: builder's 7-var raw .npz (u,v,w,theta,pressure,qv,TKE_0 -> p_prime,q_v,e_sgs).
STATE_VARS: tuple[str, ...] = ("u", "v", "w", "theta", "p_prime", "q_v", "e_sgs")
COORD_NAMES: tuple[str, ...] = ("x", "y", "z", "t")
FACE_NAMES: tuple[str, ...] = ("west", "east", "south", "north")

#: Bottom-face flux columns in the 'surface' member, after its x,y,t columns.
FLUX_VARS: tuple[str, ...] = ("fricVel", "htFlux")

RAW_MEMBERS: tuple[str, ...] = (
    "initial", "boundary", "boundary_coords", "terrain", "interior", "targets",
    "times", "surface",
)


@dataclass
class _MinMax:
    """Running global min/max per state variable and per coordinate (NaN-safe)."""

    state_min: np.ndarray
    state_max: np.ndarray
    coord_min: np.ndarray
    coord_max: np.ndarray
    flux_min: np.ndarray
    flux_max: np.ndarray

    @classmethod
    def empty(cls) -> "_MinMax":
        ns, nc, nf = len(STATE_VARS), len(COORD_NAMES), len(FLUX_VARS)
        return cls(
            state_min=np.full(ns, np.nan), state_max=np.full(ns, np.nan),
            coord_min=np.full(nc, np.nan), coord_max=np.full(nc, np.nan),
            flux_min=np.full(nf, np.nan), flux_max=np.full(nf, np.nan),
        )

    def update_state(self, arr: np.ndarray) -> None:
        """arr: (..., n_state). Merge its finite min/max per variable."""
        flat = arr.reshape(-1, arr.shape[-1])
        self.state_min = np.fmin(self.state_min, np.nanmin(flat, axis=0))
        self.state_max = np.fmax(self.state_max, np.nanmax(flat, axis=0))

    def update_coord(self, arr: np.ndarray) -> None:
        flat = arr.reshape(-1, arr.shape[-1])
        self.coord_min = np.fmin(self.coord_min, np.nanmin(flat, axis=0))
        self.coord_max = np.fmax(self.coord_max, np.nanmax(flat, axis=0))

    def update_flux(self, arr: np.ndarray) -> None:
        flat = arr.reshape(-1, arr.shape[-1])
        self.flux_min = np.fmin(self.flux_min, np.nanmin(flat, axis=0))
        self.flux_max = np.fmax(self.flux_max, np.nanmax(flat, axis=0))


def _load_raw(path: Path) -> dict[str, np.ndarray]:
    """Read one raw sub-domain .npz; validate members; derive target_mask."""
    with np.load(path) as blob:
        missing = [m for m in RAW_MEMBERS if m not in blob.files]
        if missing:
            raise ValueError(f"Raw sub-domain {path} missing members: {missing}.")
        raw = {m: blob[m].astype(np.float32) for m in RAW_MEMBERS}
    return raw


def _scan_stats(raw_paths: list[Path]) -> _MinMax:
    """Pass 1: merge global min/max over every sub-domain (never holds all at once).
    An all-NaN column (e.g. surface has no z) is expected and maps to identity."""
    stats = _MinMax.empty()
    n_coord, n_state = len(COORD_NAMES), len(STATE_VARS)
    warnings.filterwarnings("ignore", r"All-NaN slice", RuntimeWarning)
    for p in raw_paths:
        raw = _load_raw(p)
        # initial = coords then state; interior = coords only; targets = state
        stats.update_coord(raw["initial"][:, :n_coord])
        stats.update_state(raw["initial"][:, n_coord:])
        stats.update_coord(raw["interior"])
        stats.update_state(raw["targets"])
        stats.update_state(raw["boundary"])              # (faces, times, len, state)
        stats.update_coord(raw["boundary_coords"])       # (faces, times, len, coord)
        stats.update_coord(_surface_coords(raw["surface"]))   # surface x,y,t
        stats.update_flux(raw["surface"][..., 3:])            # fricVel, htFlux
    return stats


def _surface_coords(surface: np.ndarray) -> np.ndarray:
    """Lift the surface member's x,y,t columns into a (..., 4) coord array (z left
    NaN, so it does not affect z stats) for the shared coord recipe."""
    flat = surface.reshape(-1, surface.shape[-1])
    coords = np.full((flat.shape[0], len(COORD_NAMES)), np.nan, dtype=np.float32)
    coords[:, 0] = flat[:, 0]        # x
    coords[:, 1] = flat[:, 1]        # y
    coords[:, 3] = flat[:, 2]        # t
    return coords


def _norm_recipe(stats: _MinMax) -> dict:
    """minmax_01 offset/scale per variable; a flat column maps to identity."""
    def recipe(lo, hi):
        span = hi - lo
        # NaN (never seen) or zero span -> identity (offset 0, scale 1)
        scale = np.where(np.isfinite(span) & (span > 0), span, 1.0)
        offset = np.where(np.isfinite(lo), lo, 0.0)
        return offset.astype(float).tolist(), scale.astype(float).tolist()

    s_off, s_scale = recipe(stats.state_min, stats.state_max)
    c_off, c_scale = recipe(stats.coord_min, stats.coord_max)
    f_off, f_scale = recipe(stats.flux_min, stats.flux_max)
    return {
        "method": "minmax_01",
        "state_vars": list(STATE_VARS), "coord_names": list(COORD_NAMES),
        "flux_vars": list(FLUX_VARS),
        "state_offset": s_off, "state_scale": s_scale,
        "coord_offset": c_off, "coord_scale": c_scale,
        "flux_offset": f_off, "flux_scale": f_scale,
    }


def _apply(arr: np.ndarray, offset: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Normalize the last axis of arr by (arr - offset) / scale."""
    return ((arr - offset) / scale).astype(np.float32)


def _normalize_case(raw: dict[str, np.ndarray], recipe: dict) -> dict[str, np.ndarray]:
    """Pass 2: apply the global recipe to one sub-domain's members."""
    n_coord = len(COORD_NAMES)
    s_off = np.array(recipe["state_offset"], dtype=np.float32)
    s_scale = np.array(recipe["state_scale"], dtype=np.float32)
    c_off = np.array(recipe["coord_offset"], dtype=np.float32)
    c_scale = np.array(recipe["coord_scale"], dtype=np.float32)
    f_off = np.array(recipe["flux_offset"], dtype=np.float32)
    f_scale = np.array(recipe["flux_scale"], dtype=np.float32)

    initial = raw["initial"].copy()
    initial[:, :n_coord] = _apply(initial[:, :n_coord], c_off, c_scale)
    initial[:, n_coord:] = _apply(initial[:, n_coord:], s_off, s_scale)

    interior = _apply(raw["interior"], c_off, c_scale)
    targets = raw["targets"]
    mask = np.isfinite(targets).astype(np.float32)
    targets = _apply(np.where(mask > 0.0, targets, s_off), s_off, s_scale)
    targets = np.where(mask > 0.0, targets, 0.0).astype(np.float32)
    boundary = _apply(raw["boundary"], s_off, s_scale)              # state recipe
    boundary_coords = _apply(raw["boundary_coords"], c_off, c_scale)  # coord recipe

    # surface: x,y,t by coord recipe (x,y,t live at coord indices 0,1,3); fluxes by flux recipe
    surface = raw["surface"].copy()
    xyt = c_off[[0, 1, 3]], c_scale[[0, 1, 3]]
    surface[..., :3] = _apply(surface[..., :3], *xyt)
    surface[..., 3:] = _apply(surface[..., 3:], f_off, f_scale)

    return {
        "initial": initial, "boundary": boundary,
        "boundary_coords": boundary_coords, "terrain": raw["terrain"],
        "interior": interior, "targets": targets, "target_mask": mask,
        "times": raw["times"], "surface": surface,
    }


def _default_ks(k: int) -> dict[str, int]:
    return {v: k for v in STATE_VARS}


def _parse_ks(spec: str) -> dict[str, int]:
    """'16' -> all vars 16; 'u=32,v=12,w=110,theta=56' -> per-variable."""
    if "=" not in spec:
        return _default_ks(int(spec))
    ks = dict(_default_ks(16))
    for item in spec.split(","):
        var, val = item.split("=")
        ks[var.strip()] = int(val)
    return ks


def generate(
    raw_dir: Path, out_dir: Path, *, test_fraction: float = 0.2, seed: int = 0,
    ks_initial: dict[str, int] | None = None,
    ks_boundary: dict[str, int] | None = None,
) -> dict[str, list[Path]]:
    """Write normalized, POD-encoded cases into out_dir/{train,test} + metadata.json.

    POD is PER SUBDOMAIN (each case carries its own modes): boundary is a time-SVD
    of each face's (n_times x face_space); initial is a spatial-SVD over z-levels.
    ks_* give modes kept per variable. Each case stores, per field/var, its modes,
    mean, and coeffs; the raw state is dropped."""
    ks_initial = ks_initial or _default_ks(4)
    ks_boundary = ks_boundary or _default_ks(4)
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    raw_paths = sorted(raw_dir.glob("*.npz"))
    if not raw_paths:
        raise FileNotFoundError(f"No raw sub-domain .npz files in {raw_dir}.")
    log.info("found %d raw sub-domains", len(raw_paths))

    recipe = _norm_recipe(_scan_stats(raw_paths))

    ids = [p.stem for p in raw_paths]
    train_ids, test_ids = _split(ids, test_fraction, seed)
    group = {i: "train" for i in train_ids} | {i: "test" for i in test_ids}
    for g in ("train", "test"):
        (out_dir / g).mkdir(parents=True, exist_ok=True)

    written: dict[str, list[Path]] = {"train": [], "test": []}
    for p in raw_paths:
        members = _encode_case(_normalize_case(_load_raw(p), recipe),
                               ks_initial, ks_boundary)
        dest = out_dir / group[p.stem] / f"{p.stem}.npz"
        np.savez(dest, **members)
        written[group[p.stem]].append(dest)
        log.info("wrote case %s -> %s", p.stem, group[p.stem])

    _write_metadata(out_dir, recipe, written, test_fraction, seed,
                    ks_initial, ks_boundary)
    return written


def _svd_encode(mat: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """POD of mat (n_samples x n_space) over its rows. Keep k modes. Returns
    (mean (n_space,), modes (n_space x k), coeffs (n_samples x k)). Decode:
    mean + coeffs @ modes.T."""
    mean = mat.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(mat - mean, full_matrices=False)
    ke = min(k, Vt.shape[0])
    modes = Vt[:ke].T                                   # (n_space, k)
    coeffs = (U[:, :ke] * S[:ke])                       # (n_samples, k)
    return (mean.ravel().astype(np.float32), modes.astype(np.float32),
            coeffs.astype(np.float32))


def _encode_case(members: dict, ks_initial: dict, ks_boundary: dict) -> dict:
    """Per-subdomain POD encode. Boundary: per face, time-SVD. Initial: spatial-SVD
    over z-levels. Store modes/mean/coeffs per field/var; drop raw state."""
    n_coord = len(COORD_NAMES)
    out = {k: members[k] for k in
           ("boundary_coords", "terrain", "interior", "targets", "target_mask",
            "times", "surface")}
    out["initial_coords"] = members["initial"][:, :n_coord]

    # --- boundary: (n_faces, n_times, face_len, n_state); time-SVD per face,var ---
    bnd = members["boundary"]
    nf, nt, fl, _ = bnd.shape
    for vj, var in enumerate(STATE_VARS):
        for face in range(nf):
            mean, modes, coeffs = _svd_encode(bnd[face, :, :, vj], ks_boundary[var])
            out[f"bnd_mean_{var}_{face}"] = mean
            out[f"bnd_modes_{var}_{face}"] = modes
            out[f"bnd_coeffs_{var}_{face}"] = coeffs

    # --- initial: reshape to (nz, nx*ny), spatial-SVD over z-levels per var ---
    init = members["initial"]
    nx = len(np.unique(init[:, 0])); ny = len(np.unique(init[:, 1]))
    nz = len(np.unique(init[:, 2]))
    out["initial_grid"] = np.array([nx, ny, nz], dtype=np.int64)
    state = init[:, n_coord:]
    for vj, var in enumerate(STATE_VARS):
        grid = state[:, vj].reshape(nx, ny, nz).transpose(2, 0, 1).reshape(nz, nx * ny)
        mean, modes, coeffs = _svd_encode(grid, ks_initial[var])
        out[f"ini_mean_{var}"] = mean
        out[f"ini_modes_{var}"] = modes
        out[f"ini_coeffs_{var}"] = coeffs
    return out


def _split(ids: list[str], test_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    order = np.random.default_rng(seed).permutation(len(ids))
    n_test = 0 if test_fraction <= 0 else max(1, round(len(ids) * test_fraction))
    test = {ids[i] for i in order[:n_test]}
    return [i for i in ids if i not in test], [i for i in ids if i in test]


def _write_metadata(out_dir: Path, recipe: dict, written: dict,
                    test_fraction: float, seed: int,
                    ks_initial: dict, ks_boundary: dict) -> None:
    meta = {
        "schema": {
            "state_vars": list(STATE_VARS), "coord_names": list(COORD_NAMES),
            "face_names": list(FACE_NAMES), "flux_vars": list(FLUX_VARS),
            "surface": "per-snapshot bottom face: columns x,y,t,fricVel,htFlux (normalized)",
            "case_definition": "one sub-domain = (spatial extent, time window)",
            "pod": "per-subdomain; boundary time-SVD per face; initial spatial-SVD "
                   "over z. Per field/var: {mean, modes, coeffs}; decode = "
                   "mean + coeffs @ modes.T.",
        },
        "split": {"test_fraction": test_fraction, "seed": seed},
        "normalization": recipe,
        "pod_modes": {"k_initial": ks_initial, "k_boundary": ks_boundary},
        "cases": {g: [p.stem for p in paths] for g, paths in written.items()},
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Generate normalized ConditionalCase blobs.")
    ap.add_argument("raw_dir", type=Path, help="Folder of raw sub-domain .npz files.")
    ap.add_argument("out_dir", type=Path, help="Output folder (train/ test/ + metadata.json).")
    ap.add_argument("--test-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k-initial", default="16",
                    help="POD modes for initial: one int for all, or per-var "
                         "'u=32,v=12,w=110,theta=56'.")
    ap.add_argument("--k-boundary", default="16",
                    help="POD modes for boundary (same format as --k-initial).")
    ap.add_argument("-v", "--verbose", action="count", default=0)
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=[logging.WARNING, logging.INFO, logging.DEBUG][min(args.verbose, 2)],
        format="%(levelname)s %(name)s: %(message)s",
    )
    written = generate(args.raw_dir, args.out_dir,
                       test_fraction=args.test_fraction, seed=args.seed,
                       ks_initial=_parse_ks(args.k_initial),
                       ks_boundary=_parse_ks(args.k_boundary))
    print(f"Wrote {len(written['train'])} train + {len(written['test'])} test cases "
          f"+ metadata.json to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
