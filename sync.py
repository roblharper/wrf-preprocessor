"""Syncing stage: bundle each HRRR snapshot with the data that belongs to it.

A case is one HRRR snapshot. This stage groups all assembled rows into
per-snapshot bundles:

  * The HRRR rows are split into snapshots by their ``t`` value (one HRRR file
    holds many forecast times; each distinct time is a snapshot).
  * For every snapshot, LES / sensor rows are kept only if they fall within the
    time-tolerance window of the snapshot time AND inside the snapshot's spatial
    bounding box (its x/y extent). Rows matching no snapshot are dropped.

The result is one array per snapshot: HRRR interior + the co-located, co-temporal
LES / sensor rows, ready to normalize and write.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from config import (
    ANCHOR_SOURCE,
    COLUMN_INDEX,
    TIME_TOLERANCE_SECONDS,
)

_T = COLUMN_INDEX["t"]
_X = COLUMN_INDEX["x"]
_Y = COLUMN_INDEX["y"]


@dataclass
class Snapshot:
    """One HRRR snapshot and the data synced to it."""

    time: float                              # snapshot time (canonical t)
    bbox: tuple[float, float, float, float]  # x_min, x_max, y_min, y_max
    blocks: list[np.ndarray] = field(default_factory=list)

    def rows(self) -> np.ndarray:
        return np.vstack(self.blocks)


def _snapshot_id(time: float) -> str:
    """Stable, filename-safe id for a snapshot time (epoch seconds)."""

    return f"hrrr_t{int(round(time))}"


def build_snapshots(anchor_blocks: list[np.ndarray]) -> dict[str, Snapshot]:
    """Split HRRR (anchor) rows into per-snapshot bundles keyed by snapshot id.

    Each distinct HRRR ``t`` becomes a snapshot; its bounding box is the x/y
    extent of that snapshot's own rows.
    """

    if not anchor_blocks:
        raise RuntimeError(
            f"No '{ANCHOR_SOURCE}' rows found; a case needs an HRRR snapshot to "
            f"anchor it. (Is the '{ANCHOR_SOURCE}/' input folder present?)"
        )
    hrrr = np.vstack(anchor_blocks)

    snapshots: dict[str, Snapshot] = {}
    for t in np.unique(hrrr[:, _T]):
        rows = hrrr[hrrr[:, _T] == t]
        bbox = (
            float(np.nanmin(rows[:, _X])), float(np.nanmax(rows[:, _X])),
            float(np.nanmin(rows[:, _Y])), float(np.nanmax(rows[:, _Y])),
        )
        snap = Snapshot(time=float(t), bbox=bbox, blocks=[rows])
        snapshots[_snapshot_id(float(t))] = snap
    return snapshots


def _within_bbox(rows: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    x_min, x_max, y_min, y_max = bbox
    x, y = rows[:, _X], rows[:, _Y]
    return (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)


def attach_data(
    snapshots: dict[str, Snapshot],
    data_blocks: list[np.ndarray],
    *,
    tolerance_s: float = TIME_TOLERANCE_SECONDS,
) -> tuple[int, int]:
    """Attach LES / sensor rows to the snapshots they match; drop the rest.

    A row matches a snapshot when its time is within ``tolerance_s`` of the
    snapshot time and its (x, y) falls inside the snapshot bbox. A row may match
    more than one snapshot (overlapping windows); it is attached to each.

    Returns ``(kept, dropped)`` row counts for reporting / test assertions.
    """

    kept = dropped = 0
    for block in data_blocks:
        if block.size == 0:
            continue
        matched_any = np.zeros(len(block), dtype=bool)
        for snap in snapshots.values():
            in_time = np.abs(block[:, _T] - snap.time) <= tolerance_s
            in_space = _within_bbox(block, snap.bbox)
            keep = in_time & in_space
            if keep.any():
                snap.blocks.append(block[keep])
                matched_any |= keep
        kept += int(matched_any.sum())
        dropped += int((~matched_any).sum())
    return kept, dropped
