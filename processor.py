"""Processing stage: canonical assembly + normalization.

Turns reader chunks into rows in the exact canonical schema, then normalizes.
Two responsibilities, both kept behind seams so the rough-draft choices are a
one-line swap once the advisor settles the real ones:

  * Assembly -- guarantee every canonical column exists and is ordered. Columns
    a source does not supply are filled with NaN for now (a "missing" marker).
    Variable adaptation (e.g. FastEddy theta -> T, deriving density) will live
    here; it is not implemented in the draft.

  * Normalization -- a :class:`Normalizer` strategy. The draft uses per-column
    max scaling computed over the consolidated data. The domain expert wants
    HRRR-anchored stats eventually; swapping the strategy is the only change.
"""

from __future__ import annotations

from typing import Iterable, Protocol

import numpy as np

from config import CANONICAL_COLUMNS
from reader import Chunk


def assemble(chunk: Chunk) -> np.ndarray:
    """Stack a chunk into a 2-D array in canonical column order.

    Missing columns are filled with NaN (the draft's "not supplied" marker), so
    every source yields the same width and column meaning regardless of what it
    natively provides.
    """

    if not chunk:
        return np.empty((0, len(CANONICAL_COLUMNS)), dtype=np.float64)

    n_rows = len(next(iter(chunk.values())))
    columns = []
    for name in CANONICAL_COLUMNS:
        if name in chunk:
            columns.append(np.asarray(chunk[name], dtype=np.float64))
        else:
            columns.append(np.full(n_rows, np.nan, dtype=np.float64))
    return np.column_stack(columns)


# --- normalization strategy --------------------------------------------------
class Normalizer(Protocol):
    """A reversible per-column scaling, fit on data then applied to it.

    Implementations must record enough state to (a) apply the same transform to
    any chunk and (b) report the recipe for the metadata sidecar, so predictions
    can be mapped back to physical units later.
    """

    def fit(self, data: np.ndarray) -> None:
        ...

    def transform(self, data: np.ndarray) -> np.ndarray:
        ...

    def recipe(self) -> dict:
        """Serializable description of the transform (for the metadata sidecar)."""
        ...


class MaxNormalizer:
    """Draft normalizer: divide each column by its max absolute value.

    A placeholder, chosen for simplicity, not correctness -- it ignores sign and
    offset, which is wrong for signed/offset fields (u, v, w, T). It exists so
    the pipeline runs end to end; the real (HRRR-anchored, affine) scheme drops
    in by replacing this class. ``abs`` max keeps signed columns in [-1, 1].
    """

    def __init__(self) -> None:
        self._scale: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> None:
        scale = np.nanmax(np.abs(data), axis=0)
        # Guard zero/NaN columns so we never divide by zero or propagate NaN scale.
        scale = np.where(np.isfinite(scale) & (scale != 0.0), scale, 1.0)
        self._scale = scale

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self._scale is None:
            raise RuntimeError("MaxNormalizer.transform called before fit().")
        return data / self._scale

    def recipe(self) -> dict:
        if self._scale is None:
            raise RuntimeError("MaxNormalizer.recipe called before fit().")
        return {
            "method": "per_column_abs_max",
            "columns": list(CANONICAL_COLUMNS),
            "scale": self._scale.tolist(),
            "formula": "normalized = value / scale",
        }


def fit_normalizer(blocks: Iterable[np.ndarray], normalizer: Normalizer) -> Normalizer:
    """Fit a normalizer globally, over all synced rows, for use on every case.

    Per the meeting decision, normalization is global: one recipe computed over
    the consolidated data across all snapshots and sources, then applied
    uniformly so every case shares one scale. (This runs after syncing, so only
    rows that survived the time/space match contribute.)
    """

    consolidated = [b for b in blocks if b.size]
    if not consolidated:
        raise RuntimeError("No rows to fit the normalizer on after syncing.")
    normalizer.fit(np.vstack(consolidated))
    return normalizer
