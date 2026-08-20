"""Bundle each HRRR snapshot with the LES/sensor rows that belong to it.

A row belongs to a snapshot if it is within the time window and inside the
snapshot's x/y bbox; rows matching no snapshot are dropped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from config import COLUMN_INDEX, TIME_TOLERANCE_SECONDS

log = logging.getLogger(__name__)

_T = COLUMN_INDEX["t"]
_X = COLUMN_INDEX["x"]
_Y = COLUMN_INDEX["y"]


@dataclass
class Snapshot:
    """One HRRR snapshot and the data synced to it."""

    time: float
    bbox: tuple[float, float, float, float]  # x_min, x_max, y_min, y_max
    blocks: list[np.ndarray] = field(default_factory=list)

    def rows(self) -> np.ndarray:
        return np.vstack(self.blocks)


def build_snapshots(anchor_blocks: list[np.ndarray]) -> dict[str, Snapshot]:
    """One snapshot per distinct HRRR time; bbox = that time's x/y extent."""
    if not anchor_blocks:
        raise RuntimeError("No anchor (HRRR) rows found; a case needs an HRRR snapshot.")
    hrrr = np.vstack(anchor_blocks)

    snapshots: dict[str, Snapshot] = {}
    for t in np.unique(hrrr[:, _T]):
        rows = hrrr[hrrr[:, _T] == t]
        bbox = (
            float(np.nanmin(rows[:, _X])), float(np.nanmax(rows[:, _X])),
            float(np.nanmin(rows[:, _Y])), float(np.nanmax(rows[:, _Y])),
        )
        snapshots[f"hrrr_t{int(round(t))}"] = Snapshot(float(t), bbox, [rows])
    return snapshots


def attach_data(
    snapshots: dict[str, Snapshot],
    data_blocks: list[np.ndarray],
    *,
    tolerance_s: float = TIME_TOLERANCE_SECONDS,
) -> tuple[int, int]:
    """Attach matching rows to each snapshot; return (kept, dropped) counts.

    A row may match several snapshots (overlapping windows) and is attached to
    each.
    """
    kept = dropped = 0
    for block in data_blocks:
        if block.size == 0:
            continue
        matched_any = np.zeros(len(block), dtype=bool)
        for snap in snapshots.values():
            keep = (np.abs(block[:, _T] - snap.time) <= tolerance_s) & _within_bbox(block, snap.bbox)
            if keep.any():
                snap.blocks.append(block[keep])
                matched_any |= keep
        block_kept = int(matched_any.sum())
        kept += block_kept
        dropped += len(block) - block_kept
        log.debug("block of %d row(s): %d synced, %d dropped",
                  len(block), block_kept, len(block) - block_kept)
    return kept, dropped


def _within_bbox(rows: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    x_min, x_max, y_min, y_max = bbox
    x, y = rows[:, _X], rows[:, _Y]
    return (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
