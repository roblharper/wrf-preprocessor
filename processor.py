"""Canonical assembly (with derive hooks) and normalization."""

from __future__ import annotations

import logging
import warnings
from typing import Iterable

import numpy as np

from config import (
    CANONICAL_COLUMNS, COLUMN_INDEX, PHYSICAL_COLUMNS, REQUIRED_COLUMNS, SourceType,
)
from reader import Chunk
from device import array_module, make_empty, to_numpy as _to_host, to_device as to_device_arr

log = logging.getLogger(__name__)

_REQUIRED_IDX = [COLUMN_INDEX[c] for c in REQUIRED_COLUMNS]
#: optional physical columns (may be NaN for a source that does not measure them)
_OPTIONAL = tuple(c for c in PHYSICAL_COLUMNS if c not in REQUIRED_COLUMNS)


_ANCHOR_IDX = [COLUMN_INDEX[c] for c in ("t", "x", "y")]


def to_canonical(chunk: Chunk, source: SourceType, device: str = "cpu") -> np.ndarray:
    """Run the source's derive hook, stack into canonical order, drop unusable rows.
    Rows are dropped only if a required column (x,y,z,t) is NaN. ``device`` selects
    numpy (cpu) or torch (gpu) for the array math."""
    xp = array_module(device)
    if not source.canonical or not chunk:
        return make_empty(device)((0, len(CANONICAL_COLUMNS)))
    if source.derive is not None:
        chunk = {**chunk, **source.derive(chunk, xp)}
    rows = _assemble(chunk, source.source_code, make_empty(device))
    return _drop_unusable_rows(rows, source, xp)


def anchor_points(chunk: Chunk, source: SourceType) -> np.ndarray:
    """(t, x, y) triples from an anchor (HRRR); it defines snapshots, not data.

    The anchor only supplies each snapshot's time and x/y extent, so it needs no
    z or physical fields and never becomes a row in the .npy.
    """
    if not chunk:
        return np.empty((0, 3), dtype=np.float64)
    rows = _assemble(chunk, source.source_code)[:, _ANCHOR_IDX]
    return rows[np.isfinite(rows).all(axis=1)]


def _drop_unusable_rows(rows: np.ndarray, source: SourceType, xp=np) -> np.ndarray:
    """Drop rows with any required column non-finite. ``xp`` is numpy (CPU) or
    torch (GPU); the ops used are common to both."""
    if rows.shape[0] == 0:
        return rows

    missing = [c for c in _OPTIONAL if bool(xp.isnan(rows[:, COLUMN_INDEX[c]]).all())]
    if missing:
        log.info("%s: no %s (kept as NaN)", source.name, ", ".join(missing))

    keep = xp.isfinite(rows[:, _REQUIRED_IDX[0]])
    for idx in _REQUIRED_IDX[1:]:      # per-column: no fancy-index copy of the slice
        keep = keep & xp.isfinite(rows[:, idx])
    dropped = int((~keep).sum())
    if dropped:
        log.info("%s: dropped %d/%d rows missing a required coord/time",
                 source.name, dropped, rows.shape[0])
    return rows[keep]


def _assemble(chunk: Chunk, source_code: int, empty=None) -> np.ndarray:
    """Fill canonical columns into one pre-allocated (rows x NCOL) array; missing
    columns are NaN. ``empty(shape)`` allocates on the target device (numpy default;
    the GPU path passes a torch allocator)."""
    present = {k: v for k, v in chunk.items() if k in CANONICAL_COLUMNS}
    ncol = len(CANONICAL_COLUMNS)
    make = empty if empty is not None else (lambda shape: np.empty(shape, np.float64))
    if not present:
        return make((0, ncol))

    n_rows = len(next(iter(present.values())))
    out = make((n_rows, ncol))
    for j, name in enumerate(CANONICAL_COLUMNS):
        if name == "source":
            out[:, j] = source_code
        elif name in present:
            out[:, j] = present[name]
        else:
            out[:, j] = float("nan")
    return out


#: name -> function taking the stacked finite-column stats (min, max) and
#: returning per-column (offset, scale) for ``normalized = (value - offset) / scale``.
NORMALIZERS = {}


def _register(name):
    def deco(fn):
        NORMALIZERS[name] = fn
        return fn
    return deco


@_register("minmax_pm1")
def _minmax_pm1(col_min: np.ndarray, col_max: np.ndarray):
    """Map each column's [min, max] to [-1, 1]: offset = midpoint, scale = half-range."""
    offset = (col_max + col_min) / 2.0
    scale = (col_max - col_min) / 2.0
    return offset, scale

@_register("minmax_01")
def _minmax_01(col_min: np.ndarray, col_max: np.ndarray):
    """Map each column's [min, max] to [0, 1]."""
    offset = col_min
    scale = col_max - col_min
    return offset, scale

@_register("abs_max")
def _abs_max(col_min: np.ndarray, col_max: np.ndarray):
    """Scale-only by max absolute value (legacy): offset 0, keeps sign, in [-1, 1]."""
    offset = np.zeros_like(col_min)
    scale = np.maximum(np.abs(col_min), np.abs(col_max))
    return offset, scale


class RunningMinMax:
    """Streaming per-column NaN-aware min/max, so global stats need no full stack.

    Feed row blocks one at a time; peak memory is one block, not the whole set.
    Blocks may be numpy (cpu) or torch (gpu): the accumulator lives on the block's
    own device, so the GPU path needs no host round-trip until ``result()``.
    """

    def __init__(self, n_cols: int) -> None:
        self._n = n_cols
        self._min = None    # lazily created on the first block's device/backend
        self._max = None

    def update(self, block, xp=np) -> None:
        if block.shape[0] == 0:
            return
        # NaN-aware min/max without nanmin (torch lacks it): mask NaNs to +/-inf.
        # amin/amax are module funcs on both numpy and torch (axis vs dim kw differs).
        if xp is np:
            bmin = np.amin(np.where(np.isnan(block), np.inf, block), axis=0)
            bmax = np.amax(np.where(np.isnan(block), -np.inf, block), axis=0)
        else:
            bmin = xp.amin(xp.where(xp.isnan(block), float("inf"), block), dim=0)
            bmax = xp.amax(xp.where(xp.isnan(block), float("-inf"), block), dim=0)
        # seed from the first block (already on the right device/dtype), then fold.
        self._min = bmin if self._min is None else xp.minimum(self._min, bmin)
        self._max = bmax if self._max is None else xp.maximum(self._max, bmax)

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        if self._min is None:
            empty = np.full(self._n, np.nan)
            return empty, empty.copy()
        cmin, cmax = _to_host(self._min), _to_host(self._max)   # one small D2H (len n_cols)
        seen = np.isfinite(cmin) & np.isfinite(cmax)
        return np.where(seen, cmin, np.nan), np.where(seen, cmax, np.nan)


class Normalizer:
    """Global affine normalization: ``normalized = (value - offset) / scale``.

    The inverse ``value = offset + scale * normalized`` matches the codebase's
    scaling convention. The scheme is swappable via ``method`` (see NORMALIZERS);
    ``minmax_pm1`` (default) maps each column to [-1, 1].
    """

    def __init__(self, offset: np.ndarray, scale: np.ndarray, method: str) -> None:
        self._offset = offset
        self._scale = scale
        self._method = method

    @classmethod
    def from_stats(cls, col_min: np.ndarray, col_max: np.ndarray, *,
                   method: str = "minmax_pm1") -> "Normalizer":
        """Build from precomputed global min/max (see RunningMinMax)."""
        if method not in NORMALIZERS:
            raise ValueError(f"Unknown normalization method '{method}'; "
                             f"choose from {sorted(NORMALIZERS)}.")
        offset, scale = NORMALIZERS[method](col_min, col_max)
        # degenerate columns (all-NaN, or constant) become the identity map
        bad = ~(np.isfinite(offset) & np.isfinite(scale) & (scale != 0.0))
        offset = np.where(bad, 0.0, offset)
        scale = np.where(bad, 1.0, scale)
        s = COLUMN_INDEX["source"]
        offset[s], scale[s] = 0.0, 1.0   # categorical tag: never scaled
        return cls(offset, scale, method)

    @classmethod
    def fit(cls, blocks: Iterable[np.ndarray], *, method: str = "minmax_pm1") -> "Normalizer":
        """Fit globally over all synced rows, streaming (one recipe for every case).

        A column that is entirely NaN (e.g. theta/p' when no source supplied it)
        gets offset 0 / scale 1 and is left untouched, so its NaNs pass through.
        """
        stats = RunningMinMax(len(CANONICAL_COLUMNS))
        seen_any = False
        for b in blocks:
            if b.size:
                stats.update(b)
                seen_any = True
        if not seen_any:
            raise RuntimeError("No rows to fit the normalizer on after syncing.")
        col_min, col_max = stats.result()
        return cls.from_stats(col_min, col_max, method=method)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self._offset) / self._scale

    def on_device(self, device: str) -> "Normalizer":
        """Return a copy whose offset/scale live on ``device`` so transform runs
        there (GPU-side normalize before the host copy). recipe() stays host-side."""
        if device == "cpu":
            return self
        off = to_device_arr(self._offset, device)
        scale = to_device_arr(self._scale, device)
        return Normalizer(off, scale, self._method)

    def recipe(self) -> dict:
        return {
            "method": self._method,
            "columns": list(CANONICAL_COLUMNS),
            "offset": self._offset.tolist(),
            "scale": self._scale.tolist(),
            "formula": "value = offset + scale * normalized",
        }
