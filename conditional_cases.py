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


def generate(
    raw_dir: Path, out_dir: Path, *, test_fraction: float = 0.2, seed: int = 0,
) -> dict[str, list[Path]]:
    """Write normalized cases into out_dir/{train,test} plus metadata.json; return
    the written paths. One sub-domain in memory at a time."""
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

    written: dict[str, list[Path]] = {"train": [], "test": []}
    for p in raw_paths:
        members = _normalize_case(_load_raw(p), recipe)
        dest = out_dir / group[p.stem] / f"{p.stem}.npz"
        np.savez(dest, **members)
        written[group[p.stem]].append(dest)
        log.info("wrote case %s -> %s", p.stem, group[p.stem])

    _write_metadata(out_dir, recipe, written, test_fraction, seed)
    return written


def _split(ids: list[str], test_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    order = np.random.default_rng(seed).permutation(len(ids))
    n_test = 0 if test_fraction <= 0 else max(1, round(len(ids) * test_fraction))
    test = {ids[i] for i in order[:n_test]}
    return [i for i in ids if i not in test], [i for i in ids if i in test]


def _write_metadata(out_dir: Path, recipe: dict, written: dict,
                    test_fraction: float, seed: int) -> None:
    meta = {
        "schema": {
            "state_vars": list(STATE_VARS), "coord_names": list(COORD_NAMES),
            "face_names": list(FACE_NAMES),
            "members": list(RAW_MEMBERS) + ["target_mask"],
            "case_definition": "one sub-domain = (spatial extent, time window)",
        },
        "split": {"test_fraction": test_fraction, "seed": seed},
        "normalization": recipe,
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
    ap.add_argument("-v", "--verbose", action="count", default=0)
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=[logging.WARNING, logging.INFO, logging.DEBUG][min(args.verbose, 2)],
        format="%(levelname)s %(name)s: %(message)s",
    )
    written = generate(args.raw_dir, args.out_dir,
                       test_fraction=args.test_fraction, seed=args.seed)
    print(f"Wrote {len(written['train'])} train + {len(written['test'])} test cases "
          f"+ metadata.json to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
