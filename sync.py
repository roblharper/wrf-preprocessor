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


#: anchor_points columns: (t, x, y)
_A_T, _A_X, _A_Y = 0, 1, 2


@dataclass
class Snapshot:
    """One HRRR snapshot condition (time + bbox) and the data synced to it.

    HRRR itself contributes no rows; ``blocks`` holds only the LES/sensor rows
    attached to this snapshot.
    """

    time: float
    bbox: tuple[float, float, float, float]  # x_min, x_max, y_min, y_max
    blocks: list[np.ndarray] = field(default_factory=list)

    def rows(self) -> np.ndarray:
        return np.vstack(self.blocks)


def build_snapshots(anchor_blocks: list[np.ndarray]) -> dict[str, Snapshot]:
    """One snapshot per distinct HRRR time; bbox = that time's x/y extent.

    ``anchor_blocks`` are (t, x, y) triples from the anchor (HRRR). Each snapshot
    is seeded empty: HRRR defines the case condition, it is not data.
    """
    if not anchor_blocks:
        raise RuntimeError("No anchor (HRRR) rows found; a case needs an HRRR snapshot.")
    hrrr = np.vstack(anchor_blocks)

    snapshots: dict[str, Snapshot] = {}
    for t in np.unique(hrrr[:, _A_T]):
        pts = hrrr[hrrr[:, _A_T] == t]
        bbox = (
            float(np.nanmin(pts[:, _A_X])), float(np.nanmax(pts[:, _A_X])),
            float(np.nanmin(pts[:, _A_Y])), float(np.nanmax(pts[:, _A_Y])),
        )
        snapshots[f"hrrr_t{int(round(t))}"] = Snapshot(float(t), bbox)
    return snapshots


class SnapshotIndex:
    """Snapshots indexed by time so a block only checks those in its time window.

    Testing every block against all snapshots is O(blocks x snapshots) over the
    full row count, which dominates at scale (e.g. 744 cases x 4.7M rows). Each
    block spans one narrow time range, so a sorted-time bisection limits the
    check to the handful of snapshots within +/- tolerance.
    """

    def __init__(self, snapshots: dict[str, Snapshot], *,
                 tolerance_s: float = TIME_TOLERANCE_SECONDS) -> None:
        self._tol = tolerance_s
        items = sorted(snapshots.items(), key=lambda kv: kv[1].time)
        self._ids = [cid for cid, _ in items]
        self._snaps = [s for _, s in items]
        self._times = np.array([s.time for s in self._snaps], dtype=np.float64)

    def candidates(self, t_lo: float, t_hi: float):
        """Snapshots whose time could match rows in [t_lo, t_hi] within tolerance."""
        import bisect
        lo = bisect.bisect_left(self._times, t_lo - self._tol)
        hi = bisect.bisect_right(self._times, t_hi + self._tol)
        return zip(self._ids[lo:hi], self._snaps[lo:hi])


def match_snapshots(
    block: np.ndarray,
    snapshots,
    *,
    tolerance_s: float = TIME_TOLERANCE_SECONDS,
    xp=np,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Split one block into per-snapshot matching rows, without holding state.

    ``snapshots`` may be a dict or a ``SnapshotIndex``; a row may match several
    snapshots and appears in each. ``xp`` is numpy (CPU) or torch (GPU)."""
    index = snapshots if isinstance(snapshots, SnapshotIndex) \
        else SnapshotIndex(snapshots, tolerance_s=tolerance_s)
    t = block[:, _T]
    t_lo, t_hi = float(t.min()), float(t.max())   # one sync per block, not per candidate
    matched_any = xp.isfinite(t) & False   # all-False mask on t's device/backend
    per_case: dict[str, np.ndarray] = {}
    for cid, snap in index.candidates(t_lo, t_hi):
        keep = (xp.abs(t - snap.time) <= tolerance_s) & _within_bbox(block, snap.bbox)
        per_case[cid] = block[keep]        # keep the gather; may be empty (no sync)
        matched_any = matched_any | keep
    return {cid: rows for cid, rows in per_case.items() if rows.shape[0]}, matched_any


def attach_data(
    snapshots: dict[str, Snapshot],
    data_blocks: list[np.ndarray],
    *,
    tolerance_s: float = TIME_TOLERANCE_SECONDS,
) -> tuple[int, int]:
    """Attach matching rows to each snapshot in memory; return (kept, dropped).

    In-memory path kept for tests/small runs; the pipeline streams via
    ``match_snapshots`` instead. A row may match several snapshots.
    """
    kept = dropped = 0
    for block in data_blocks:
        if block.size == 0:
            continue
        per_case, matched_any = match_snapshots(block, snapshots, tolerance_s=tolerance_s)
        for cid, rows in per_case.items():
            snapshots[cid].blocks.append(rows)
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
