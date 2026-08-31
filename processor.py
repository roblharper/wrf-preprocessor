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

log = logging.getLogger(__name__)

_REQUIRED_IDX = [COLUMN_INDEX[c] for c in REQUIRED_COLUMNS]
#: optional physical columns (may be NaN for a source that does not measure them)
_OPTIONAL = tuple(c for c in PHYSICAL_COLUMNS if c not in REQUIRED_COLUMNS)


_ANCHOR_IDX = [COLUMN_INDEX[c] for c in ("t", "x", "y")]


def to_canonical(chunk: Chunk, source: SourceType) -> np.ndarray:
    """Run the source's derive hook, stack into canonical order, drop unusable rows.

    A row is dropped only if a REQUIRED column (x,y,z,t) is NaN. Optional columns
    (u,v,w,theta,p_prime) may be NaN for a source that does not measure them; the
    missing ones are logged, not dropped. A ``canonical=False`` source contributes
    no rows.
    """
    if not source.canonical or not chunk:
        return np.empty((0, len(CANONICAL_COLUMNS)), dtype=np.float64)
    if source.derive is not None:
        chunk = {**chunk, **source.derive(chunk)}
    rows = _assemble(chunk, source.source_code)
    return _drop_unusable_rows(rows, source)


def anchor_points(chunk: Chunk, source: SourceType) -> np.ndarray:
    """(t, x, y) triples from an anchor (HRRR); it defines snapshots, not data.

    The anchor only supplies each snapshot's time and x/y extent, so it needs no
    z or physical fields and never becomes a row in the .npy.
    """
    if not chunk:
        return np.empty((0, 3), dtype=np.float64)
    rows = _assemble(chunk, source.source_code)[:, _ANCHOR_IDX]
    return rows[np.isfinite(rows).all(axis=1)]


def _drop_unusable_rows(rows: np.ndarray, source: SourceType) -> np.ndarray:
    """Drop rows missing a required column; log optional columns the source lacks."""
    if rows.size == 0:
        return rows

    missing = [c for c in _OPTIONAL if np.isnan(rows[:, COLUMN_INDEX[c]]).all()]
    if missing:
        log.info("%s: no %s (kept as NaN)", source.name, ", ".join(missing))

    keep = np.isfinite(rows[:, _REQUIRED_IDX]).all(axis=1)
    dropped = int((~keep).sum())
    if dropped:
        log.info("%s: dropped %d/%d rows missing a required coord/time",
                 source.name, dropped, len(rows))
    return rows[keep]


def _assemble(chunk: Chunk, source_code: int) -> np.ndarray:
    """Stack canonical columns into a 2-D array; missing physical columns are NaN.

    The ``source`` column is filled with ``source_code`` for every row.
    """
    present = {k: v for k, v in chunk.items() if k in CANONICAL_COLUMNS}
    if not present:
        return np.empty((0, len(CANONICAL_COLUMNS)), dtype=np.float64)

    n_rows = len(next(iter(present.values())))
    columns = []
    for name in CANONICAL_COLUMNS:
        if name == "source":
            columns.append(np.full(n_rows, source_code, dtype=np.float64))
        elif name in present:
            columns.append(np.asarray(present[name], dtype=np.float64))
        else:
            columns.append(np.full(n_rows, np.nan, dtype=np.float64))
    return np.column_stack(columns)


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
    def fit(cls, blocks: Iterable[np.ndarray], *, method: str = "minmax_pm1",
            bounds: dict[str, tuple[float, float]] | None = None,) -> "Normalizer":
        """Fit globally over all synced rows (one recipe for every case).

        A column that is entirely NaN (e.g. theta/p' when no source supplied it)
        gets offset 0 / scale 1 and is left untouched, so its NaNs pass through.
        """
        if method not in NORMALIZERS:
            raise ValueError(f"Unknown normalization method '{method}'; "
                             f"choose from {sorted(NORMALIZERS)}.")
        consolidated = [b for b in blocks if b.size]
        if not consolidated:
            raise RuntimeError("No rows to fit the normalizer on after syncing.")
        stacked = np.vstack(consolidated)
        with warnings.catch_warnings():   # all-NaN columns are expected (optional cols)
            warnings.simplefilter("ignore", RuntimeWarning)
            col_min = np.nanmin(stacked, axis=0)
            col_max = np.nanmax(stacked, axis=0)

        if bounds:
            for column, (lower, upper) in bounds.items():
                if column not in COLUMN_INDEX:
                    raise ValueError(f"Unknown normalization column: {column}")
                if upper <= lower:
                    raise ValueError(f"Invalid normalization bounds for {column}:"
                        f" {lower}, {upper}")

                index = COLUMN_INDEX[column]
                col_min[index] = lower
                col_max[index] = upper

        offset, scale = NORMALIZERS[method](col_min, col_max)
        # degenerate columns (all-NaN, or constant) become the identity map
        bad = ~(np.isfinite(offset) & np.isfinite(scale) & (scale != 0.0))
        offset = np.where(bad, 0.0, offset)
        scale = np.where(bad, 1.0, scale)
        s = COLUMN_INDEX["source"]
        offset[s], scale[s] = 0.0, 1.0   # categorical tag: never scaled
        return cls(offset, scale, method)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self._offset) / self._scale

    def recipe(self) -> dict:
        return {
            "method": self._method,
            "columns": list(CANONICAL_COLUMNS),
            "offset": self._offset.tolist(),
            "scale": self._scale.tolist(),
            "formula": "value = offset + scale * normalized",
        }
