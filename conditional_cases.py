"""Normalize raw sub-domain .npz files into ConditionalCase blobs the PINN reads.

Two passes: merge one global min/max recipe over all sub-domains, then normalize
each. Reads .npz only (no netCDF); source agnostic.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

#: Variable order carried in the state columns of every member.
STATE_VARS: tuple[str, ...] = ("u", "v", "w", "theta")
COORD_NAMES: tuple[str, ...] = ("x", "y", "z", "t")
FACE_NAMES: tuple[str, ...] = ("west", "east", "south", "north")

RAW_MEMBERS: tuple[str, ...] = (
    "initial", "boundary", "boundary_coords", "terrain", "interior", "targets", "times",
)


@dataclass
class _MinMax:
    """Running global min/max per state variable and per coordinate (NaN-safe)."""

    state_min: np.ndarray
    state_max: np.ndarray
    coord_min: np.ndarray
    coord_max: np.ndarray

    @classmethod
    def empty(cls) -> "_MinMax":
        ns, nc = len(STATE_VARS), len(COORD_NAMES)
        return cls(
            state_min=np.full(ns, np.nan), state_max=np.full(ns, np.nan),
            coord_min=np.full(nc, np.nan), coord_max=np.full(nc, np.nan),
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


def _load_raw(path: Path) -> dict[str, np.ndarray]:
    """Read one raw sub-domain .npz; validate members; derive target_mask."""
    with np.load(path) as blob:
        missing = [m for m in RAW_MEMBERS if m not in blob.files]
        if missing:
            raise ValueError(f"Raw sub-domain {path} missing members: {missing}.")
        raw = {m: blob[m].astype(np.float32) for m in RAW_MEMBERS}
    return raw


def _scan_stats(raw_paths: list[Path]) -> _MinMax:
    """Pass 1: merge global min/max over every sub-domain (never holds all at once)."""
    stats = _MinMax.empty()
    n_coord, n_state = len(COORD_NAMES), len(STATE_VARS)
    for p in raw_paths:
        raw = _load_raw(p)
        # initial = coords then state; interior = coords only; targets = state
        stats.update_coord(raw["initial"][:, :n_coord])
        stats.update_state(raw["initial"][:, n_coord:])
        stats.update_coord(raw["interior"])
        stats.update_state(raw["targets"])
        stats.update_state(raw["boundary"])              # (faces, times, len, state)
        stats.update_coord(raw["boundary_coords"])       # (faces, times, len, coord)
    return stats


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
    return {
        "method": "minmax_01",
        "state_vars": list(STATE_VARS), "coord_names": list(COORD_NAMES),
        "state_offset": s_off, "state_scale": s_scale,
        "coord_offset": c_off, "coord_scale": c_scale,
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

    return {
        "initial": initial, "boundary": boundary,
        "boundary_coords": boundary_coords, "terrain": raw["terrain"],
        "interior": interior, "targets": targets, "target_mask": mask,
        "times": raw["times"],
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
    """Write normalized, POD-encoded cases into out_dir/{train,test} plus
    metadata.json and pod_modes.npz. Each case stores per-variable coeffs z (not
    raw state) for initial and boundary; ks_* give the modes kept per variable."""
    ks_initial = ks_initial or _default_ks(16)
    ks_boundary = ks_boundary or _default_ks(16)
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    raw_paths = sorted(raw_dir.glob("*.npz"))
    if not raw_paths:
        raise FileNotFoundError(f"No raw sub-domain .npz files in {raw_dir}.")
    log.info("found %d raw sub-domains", len(raw_paths))

    stats = _scan_stats(raw_paths)
    recipe = _norm_recipe(stats)

    ids = [p.stem for p in raw_paths]
    train_ids, test_ids = _split(ids, test_fraction, seed)
    group = {i: "train" for i in train_ids} | {i: "test" for i in test_ids}
    for g in ("train", "test"):
        (out_dir / g).mkdir(parents=True, exist_ok=True)

    # POD basis is built from the TRAINING cases only, so held-out reconstruction
    # is a real test. Only the (small) state columns are stacked, not full cases.
    train_members = [_normalize_case(_load_raw(raw_dir / f"{i}.npz"), recipe)
                     for i in train_ids]
    pod = _build_pod(train_members, ks_initial, ks_boundary)
    log.info("POD modes/var: initial %s, boundary %s",
             pod["initial"]["k"], pod["boundary"]["k"])

    written: dict[str, list[Path]] = {"train": [], "test": []}
    for p in raw_paths:
        members = _encode_case(_normalize_case(_load_raw(p), recipe), pod)
        dest = out_dir / group[p.stem] / f"{p.stem}.npz"
        np.savez(dest, **members)
        written[group[p.stem]].append(dest)
        log.info("wrote case %s -> %s", p.stem, group[p.stem])

    np.savez(out_dir / "pod_modes.npz", **_flatten_pod(pod))
    _write_metadata(out_dir, recipe, written, test_fraction, seed, pod)
    return written


def _flatten_pod(pod: dict) -> dict:
    """Flatten the nested per-variable POD dict to arrays for npz: keys like
    'initial_mean_u', 'initial_modes_u', plus 'initial_n_points'."""
    flat = {}
    for field in ("initial", "boundary"):
        flat[f"{field}_n_points"] = np.int64(pod[field]["n_points"])
        for var in STATE_VARS:
            flat[f"{field}_mean_{var}"] = pod[field]["means"][var]
            flat[f"{field}_modes_{var}"] = pod[field]["modes"][var]
    return flat


def _encode_case(members: dict, pod: dict) -> dict:
    """Replace raw initial/boundary state with per-variable POD coeffs z; keep
    coords. Interior, targets, terrain, times pass through unchanged."""
    n_coord = len(COORD_NAMES)
    out = dict(members)
    out["initial_coords"] = members["initial"][:, :n_coord]
    out["z_initial"] = _encode_field(members["initial"][:, n_coord:], pod["initial"])
    bnd_state = members["boundary"].reshape(-1, len(STATE_VARS))
    out["z_boundary"] = _encode_field(bnd_state, pod["boundary"])
    del out["initial"], out["boundary"]          # raw state dropped
    return out


def _var_basis(column: list[np.ndarray], k: int) -> tuple[np.ndarray, np.ndarray]:
    """POD basis for one variable's column across training cases. column: list of
    (n_points,) vectors. Returns mean (n_points,) and modes V (n_points, k_eff)."""
    X = np.stack(column, axis=1).astype(np.float64)       # (n_points, n_cases)
    mean = X.mean(axis=1, keepdims=True)
    U, _, _ = np.linalg.svd(X - mean, full_matrices=False)
    k_eff = min(k, U.shape[1])
    return mean.ravel().astype(np.float32), U[:, :k_eff].astype(np.float32)


def _build_pod_field(fields: list[np.ndarray], ks: dict[str, int]) -> dict:
    """Per-variable POD for one field. fields: list of (n_points, n_state). ks maps
    each state var to its mode count. Returns per-var means, modes, and k_eff."""
    out = {"means": {}, "modes": {}, "k": {}, "n_points": int(fields[0].shape[0])}
    for j, var in enumerate(STATE_VARS):
        col = [f[:, j] for f in fields]
        mean, V = _var_basis(col, ks[var])
        out["means"][var] = mean
        out["modes"][var] = V
        out["k"][var] = int(V.shape[1])
    return out


def _encode_field(field: np.ndarray, basis: dict) -> np.ndarray:
    """z = concat over variables of V_var^T (col - mean_var). Returns flat (sum k,)."""
    parts = []
    for j, var in enumerate(STATE_VARS):
        parts.append(basis["modes"][var].T @ (field[:, j] - basis["means"][var]))
    return np.concatenate(parts).astype(np.float32)


def _build_pod(train_members: list[dict], ks_initial: dict, ks_boundary: dict) -> dict:
    """Per-variable global POD bases for the initial and boundary state fields."""
    n_coord = len(COORD_NAMES)
    init_fields = [m["initial"][:, n_coord:] for m in train_members]
    bnd_fields = [m["boundary"].reshape(-1, len(STATE_VARS)) for m in train_members]
    return {"initial": _build_pod_field(init_fields, ks_initial),
            "boundary": _build_pod_field(bnd_fields, ks_boundary)}


def _split(ids: list[str], test_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    order = np.random.default_rng(seed).permutation(len(ids))
    n_test = 0 if test_fraction <= 0 else max(1, round(len(ids) * test_fraction))
    test = {ids[i] for i in order[:n_test]}
    return [i for i in ids if i not in test], [i for i in ids if i in test]


def _write_metadata(out_dir: Path, recipe: dict, written: dict,
                    test_fraction: float, seed: int, pod: dict) -> None:
    meta = {
        "schema": {
            "state_vars": list(STATE_VARS), "coord_names": list(COORD_NAMES),
            "face_names": list(FACE_NAMES),
            "members": ["initial_coords", "z_initial", "z_boundary",
                        "boundary_coords", "terrain", "interior", "targets",
                        "target_mask", "times"],
            "case_definition": "one sub-domain = (spatial extent, time window)",
        },
        "split": {"test_fraction": test_fraction, "seed": seed},
        "normalization": recipe,
        "pod": {
            "modes_file": "pod_modes.npz",
            "per_variable": True,
            "k_initial": pod["initial"]["k"],
            "k_boundary": pod["boundary"]["k"],
            "initial_n_points": pod["initial"]["n_points"],
            "boundary_n_points": pod["boundary"]["n_points"],
        },
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
