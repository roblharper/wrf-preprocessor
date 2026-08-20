"""End-to-end test: phony data in, synced per-snapshot .npy out."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Make the preprocessor modules + fixtures importable regardless of CWD.
_PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PKG))
sys.path.insert(0, str(_PKG / "fixtures"))

from config import COLUMN_INDEX, TIME_TOLERANCE_SECONDS
from reader import discover_files, read_file
from processor import to_canonical
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
    """Re-run read -> canonical -> sync on raw (un-normalized) rows."""

    anchor, data = [], []
    for path, src in discover_files(root):
        for ch in read_file(path, src, chunk_size=50):
            b = to_canonical(ch, src)
            if b.size:
                (anchor if src.is_anchor else data).append(b)
    snaps = build_snapshots(anchor)
    kept, dropped = attach_data(snaps, data)
    return snaps, kept, dropped


def test_pipeline_writes_one_npy_per_snapshot(phony, tmp_path):
    root, manifest = phony
    out = tmp_path / "out"
    written = orchestrator.run(root, out, chunk_size=50)

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
    orchestrator.run(root, out, chunk_size=50)

    meta = json.loads((out / "metadata.json").read_text())
    # one global recipe with a scale per canonical column
    scale = meta["normalization"]["scale"]
    assert len(scale) == len(COLUMN_INDEX)

    # every written case is normalized by that same scale -> values within [-1, 1]
    for name in meta["cases"]:
        d = np.load(out / f"{name}.npy")
        finite = d[np.isfinite(d)]
        assert np.all(np.abs(finite) <= 1.0 + 1e-9), f"{name}: values exceed global scale"


# --- multi-type registry: discovery, derive hooks, unmapped types ------------
def test_discovery_matches_multiple_instrument_types(phony):
    """Every structural type is discovered by filename, across a mixed folder."""

    root, _ = phony
    matched = {src.match for _, src in discover_files(root)}
    # at least the anchor, interior, and several observation types are present
    for expected in ("hrrr", "les", "ecorsfwind", "smos", "twr", "co2flx",
                     "armbeatm", "dlaux"):
        assert expected in matched, f"{expected} not discovered"


def test_derive_hooks_produce_correct_canonical_values(phony):
    """ARM time (base+offset), smos speed/dir -> u,v, and twr Celsius -> Kelvin."""

    root, _ = phony
    obs = root / "obs"

    def first_canonical(match_glob):
        from config import match_source
        p = next((obs).glob(match_glob))
        src = match_source(p.name)
        for ch in read_file(p, src, chunk_size=50):
            b = to_canonical(ch, src)
            if b.size:
                return b
        return None

    # ARM absolute time is epoch-scale, not seconds-since-midnight.
    ecor = first_canonical("ecorsfwind_a*")
    assert ecor[0, _T] > 1_000_000_000, "ecor t is not an absolute epoch time"

    # smos: wspd=5, wdir=270 (from the west) -> u ~ +5, v ~ 0.
    U, V = COLUMN_INDEX["u"], COLUMN_INDEX["v"]
    smos = first_canonical("smos_a*")
    assert abs(smos[0, U] - 5.0) < 0.5 and abs(smos[0, V]) < 0.5

    # twr: 18 C -> ~291 K.
    Tc = COLUMN_INDEX["T"]
    twr = first_canonical("twr25m*")
    assert 288.0 < twr[0, Tc] < 294.0


def test_unmapped_type_contributes_no_rows(phony):
    """A canonical=False type (dlaux) is recognized but adds no canonical rows."""

    from config import match_source
    root, _ = phony
    p = next((root / "obs").glob("dlaux_a*"))
    src = match_source(p.name)
    assert src is not None and src.canonical is False
    rows = [to_canonical(ch, src) for ch in read_file(p, src, chunk_size=50)]
    assert all(b.shape[0] == 0 for b in rows), "unmapped type produced rows"
