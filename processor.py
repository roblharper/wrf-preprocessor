"""Canonical assembly (with derive hooks) and normalization."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from config import CANONICAL_COLUMNS, SourceType
from reader import Chunk


def to_canonical(chunk: Chunk, source: SourceType) -> np.ndarray:
    """Run the source's derive hook, then stack into canonical column order.

    A ``canonical=False`` source contributes no rows.
    """
    if not source.canonical or not chunk:
        return np.empty((0, len(CANONICAL_COLUMNS)), dtype=np.float64)
    if source.derive is not None:
        chunk = {**chunk, **source.derive(chunk)}
    return _assemble(chunk)


def _assemble(chunk: Chunk) -> np.ndarray:
    """Stack canonical columns into a 2-D array; missing columns are NaN."""
    present = {k: v for k, v in chunk.items() if k in CANONICAL_COLUMNS}
    if not present:
        return np.empty((0, len(CANONICAL_COLUMNS)), dtype=np.float64)

    n_rows = len(next(iter(present.values())))
    columns = [
        np.asarray(present[name], dtype=np.float64) if name in present
        else np.full(n_rows, np.nan, dtype=np.float64)
        for name in CANONICAL_COLUMNS
    ]
    return np.column_stack(columns)


class Normalizer:
    """Per-column scaling by max absolute value (keeps columns in [-1, 1]).

    A placeholder scheme; replace the body here to adopt the real one.
    """

    def __init__(self, scale: np.ndarray) -> None:
        self._scale = scale

    @classmethod
    def fit(cls, blocks: Iterable[np.ndarray]) -> "Normalizer":
        """Fit globally over all synced rows (one recipe for every case)."""
        consolidated = [b for b in blocks if b.size]
        if not consolidated:
            raise RuntimeError("No rows to fit the normalizer on after syncing.")
        scale = np.nanmax(np.abs(np.vstack(consolidated)), axis=0)
        return cls(np.where(np.isfinite(scale) & (scale != 0.0), scale, 1.0))

    def transform(self, data: np.ndarray) -> np.ndarray:
        return data / self._scale

    def recipe(self) -> dict:
        return {
            "method": "per_column_abs_max",
            "columns": list(CANONICAL_COLUMNS),
            "scale": self._scale.tolist(),
            "formula": "normalized = value / scale",
        }
