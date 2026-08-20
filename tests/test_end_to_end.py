"""End-to-end test of the pre-processor's universal syncing behavior.

Generates phony NetCDF data (HRRR / LES / sensor) with a small fraction of rows
deliberately off in time and/or space, runs the full pipeline, and asserts:

  * one .npy is written per HRRR snapshot, plus a shared metadata.json;
  * every row in a snapshot file is within the time tolerance AND spatial bbox
    of that snapshot -- i.e. off-time / off-location junk was dropped;
  * the off-location sensor station never appears in any output;
  * normalization is global (one recipe over all synced rows).

Run: pytest  (from the preprocessor/ directory)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Make the preprocessor modules importable when pytest runs from anywhere.
_PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PKG))
sys.path.insert(0, str(_PKG / "fixtures"))

from config import COLUMN_INDEX, TIME_TOLERANCE_SECONDS, source_for, ANCHOR_SOURCE
from reader import iter_source_files, read_file
from processor import assemble
from sync import build_snapshots, attach_data
import orchestrator
import generate_phony_data as gen

_T = COLUMN_INDEX["t"]
_X = COLUMN_INDEX["x"]
_Y = COLUMN_INDEX["y"]


@pytest.fixture
def phony(tmp_path):
    root = tmp_path / "in"
    manifest = gen.generate(root, seed=0)
    return root, manifest


def _raw_snapshots(root: Path):
    """Re-run the sync stage on raw (un-normalized) rows, for value assertions."""

    anchor, data = [], []
    for key in ("hrrr", "simulation", "sensor"):
        src = source_for(key)
        for p in iter_source_files(root / key):
            for ch in read_file(p, src, chunk_size=50):
                b = assemble(ch)
                if b.size:
                    (anchor if key == ANCHOR_SOURCE else data).append(b)
    snaps = build_snapshots(anchor)
    kept, dropped = attach_data(snaps, data)
    return snaps, kept, dropped


def test_pipeline_writes_one_npy_per_snapshot(phony, tmp_path):
    root, manifest = phony
    out = tmp_path / "out"
    written = orchestrator.run(root, out, chunk_size=50, verbose=False)

    # two distinct HRRR times -> two snapshots + metadata.json
    assert len(written) == 2
    assert (out / "metadata.json").exists()
    stems = {p.stem for p in written}
    assert stems == {f"hrrr_t{int(manifest['sync_time'])}",
                     f"hrrr_t{int(manifest['off_time'])}"}


def test_every_row_is_time_and_space_synced(phony):
    root, _ = phony
    snaps, kept, dropped = _raw_snapshots(root)

    assert dropped > 0, "expected some off-time/off-location rows to be dropped"
    for cid, snap in snaps.items():
        r = snap.rows()
        dt = np.abs(r[:, _T] - snap.time)
        assert (dt <= TIME_TOLERANCE_SECONDS).all(), f"{cid}: a row is outside the time window"
        x_min, x_max, y_min, y_max = snap.bbox
        assert ((r[:, _X] >= x_min) & (r[:, _X] <= x_max)).all(), f"{cid}: a row is outside bbox x"
        assert ((r[:, _Y] >= y_min) & (r[:, _Y] <= y_max)).all(), f"{cid}: a row is outside bbox y"


def test_off_location_station_is_excluded(phony):
    root, manifest = phony
    snaps, _, _ = _raw_snapshots(root)

    # the far station is at lon ~ (region_center + 5 deg); it must not survive.
    far_lon_min = np.mean(manifest["region_lon"]) + 4.0
    for cid, snap in snaps.items():
        r = snap.rows()
        assert (r[:, _X] < far_lon_min).all(), f"{cid}: off-location rows leaked in"


def test_off_time_snapshot_has_no_sync_time_data(phony):
    root, manifest = phony
    snaps, _, _ = _raw_snapshots(root)

    off_id = f"hrrr_t{int(manifest['off_time'])}"
    r = snaps[off_id].rows()
    # nothing in the off-time snapshot should carry the sync-time stamp
    assert (np.abs(r[:, _T] - manifest["sync_time"]) > TIME_TOLERANCE_SECONDS).all()


def test_normalization_is_global(phony, tmp_path):
    root, _ = phony
    out = tmp_path / "out"
    orchestrator.run(root, out, chunk_size=50, verbose=False)

    meta = json.loads((out / "metadata.json").read_text())
    # one global recipe with a scale per canonical column
    scale = meta["normalization"]["scale"]
    assert len(scale) == len(COLUMN_INDEX)

    # every written case is normalized by that same scale -> values within [-1, 1]
    for name in meta["cases"]:
        d = np.load(out / f"{name}.npy")
        finite = d[np.isfinite(d)]
        assert np.all(np.abs(finite) <= 1.0 + 1e-9), f"{name}: values exceed global scale"
