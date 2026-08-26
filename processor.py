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


class Normalizer:
    """Per-column scaling by max absolute value (keeps columns in [-1, 1]).

    A placeholder scheme; replace the body here to adopt the real one.
    """

    def __init__(self, scale: np.ndarray) -> None:
        self._scale = scale

    @classmethod
    def fit(cls, blocks: Iterable[np.ndarray]) -> "Normalizer":
        """Fit globally over all synced rows (one recipe for every case).

        A column that is entirely NaN (e.g. theta/p' when no source supplied it)
        gets scale 1.0 and is left untouched, so its NaNs pass through unchanged.
        """
        consolidated = [b for b in blocks if b.size]
        if not consolidated:
            raise RuntimeError("No rows to fit the normalizer on after syncing.")
        with warnings.catch_warnings():   # all-NaN columns are expected (optional cols)
            warnings.simplefilter("ignore", RuntimeWarning)
            scale = np.nanmax(np.abs(np.vstack(consolidated)), axis=0)
        scale = np.where(np.isfinite(scale) & (scale != 0.0), scale, 1.0)
        scale[COLUMN_INDEX["source"]] = 1.0   # categorical tag: never scaled
        return cls(scale)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return data / self._scale

    def recipe(self) -> dict:
        return {
            "method": "per_column_abs_max",
            "columns": list(CANONICAL_COLUMNS),
            "scale": self._scale.tolist(),
            "formula": "normalized = value / scale",
        }
