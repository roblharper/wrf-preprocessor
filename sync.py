"""Read an HRRR file's snapshot time to tag a case.

Folder-per-case: the HRRR file no longer groups rows (no time window, no bbox).
It only records the snapshot time, kept in metadata for traceability.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from config import SourceType
from processor import anchor_points
from reader import read_file


def read_hrrr_time(path: Path, source: SourceType, *, chunk_size: int = 1) -> float:
    """Single snapshot time (epoch s) from an HRRR file: the min valid_time seen.

    HRRR tiles carry one valid_time; if a file holds several, the earliest is the
    case's snapshot time. Raises if the file yields no finite time.
    """
    times: list[float] = []
    for chunk in read_file(path, source, chunk_size=chunk_size):
        pts = anchor_points(chunk, source)   # (t, x, y); we only need t
        if pts.size:
            times.append(float(np.min(pts[:, 0])))
    if not times:
        raise RuntimeError(f"{path.name}: no finite HRRR valid_time found.")
    return min(times)
