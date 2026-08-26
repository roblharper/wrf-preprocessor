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


def test_pipeline_writes_split_cases(phony, tmp_path):
    root, manifest = phony
    out = tmp_path / "out"
    written = orchestrator.run(root, out, chunk_size=50)

    # two snapshots, split into train/ and test/ dirs + a shared metadata.json
    all_stems = {p.stem for p in written["train"] + written["test"]}
    assert all_stems == {f"hrrr_t{int(manifest['sync_time'])}",
                         f"hrrr_t{int(manifest['off_time'])}"}
    assert (out / "metadata.json").exists()
    assert len(written["test"]) >= 1  # at least one held out
    # splits are disjoint
    assert not ({p.stem for p in written["train"]} & {p.stem for p in written["test"]})


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

    # physical columns are within [-1, 1]; the source tag is left unscaled.
    phys = [i for c, i in COLUMN_INDEX.items() if c != "source"]
    for f in out.rglob("*.npy"):
        d = np.load(f)[:, phys]
        finite = d[np.isfinite(d)]
        assert np.all(np.abs(finite) <= 1.0 + 1e-9), f"{f.stem}: values exceed global scale"


# --- multi-type registry: discovery, derive hooks, unmapped types ------------
def test_discovery_matches_multiple_instrument_types(phony):
    """Every structural type is discovered by filename, across a mixed folder."""

    root, _ = phony
    matched = {src.match for _, src in discover_files(root)}
    for expected in ("hrrr", "wrfout", "ecorsfwind", "smos", "twr", "co2flx",
                     "armbeatm", "dlaux"):
        assert expected in matched, f"{expected} not discovered"


def test_derive_hooks_produce_correct_canonical_values(phony):
    """ARM time (base+offset) and smos speed/direction -> u, v."""

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

    # ecor has coordinates, so it survives the NaN-row drop; check absolute time.
    ecor = first_canonical("ecorsfwind_a*")
    assert ecor[0, _T] > 1_000_000_000, "ecor t is not an absolute epoch time"

    # smos has no lat/lon (dropped downstream), so test its wind derivation at
    # the hook level: wspd=5, wdir=270 (from the west) -> u ~ +5, v ~ 0.
    from config import _uv_from_speed_dir
    import numpy as np
    out = _uv_from_speed_dir({"wspd": np.array([5.0]), "wdir": np.array([270.0])})
    assert abs(out["u"][0] - 5.0) < 0.5 and abs(out["v"][0]) < 0.5


def test_source_tag_written_per_category(phony, tmp_path):
    """Each output row carries its source category code (inlet/sim/sensor)."""

    from config import SRC_INLET, SRC_SIM, SRC_SENSOR
    root, _ = phony
    out = tmp_path / "out"
    orchestrator.run(root, out, chunk_size=50)

    src_col = COLUMN_INDEX["source"]
    all_tags = set()
    for f in out.rglob("*.npy"):
        all_tags |= set(np.load(f)[:, src_col].astype(int).tolist())
    assert all_tags == {SRC_INLET, SRC_SIM, SRC_SENSOR}


def test_optional_columns_nan_do_not_drop_rows(phony, tmp_path):
    """Sources lacking theta/p' keep their rows (NaN); LASSO supplies theta/p'."""

    from config import SRC_SIM, SRC_SENSOR
    root, _ = phony
    out = tmp_path / "out"
    orchestrator.run(root, out, chunk_size=50)

    src = COLUMN_INDEX["source"]
    th, pp = COLUMN_INDEX["theta"], COLUMN_INDEX["p_prime"]
    rows = np.vstack([np.load(f) for f in out.rglob("*.npy")])

    sim = rows[rows[:, src] == SRC_SIM]
    sensor = rows[rows[:, src] == SRC_SENSOR]
    # LASSO (sim) supplies theta/p'; sensors keep rows but with NaN theta/p'
    assert sim.size and np.isfinite(sim[:, [th, pp]]).all()
    assert sensor.size and np.isnan(sensor[:, [th, pp]]).all()
    # required coords/time are never NaN in any kept row
    req = [COLUMN_INDEX[c] for c in ("x", "y", "z", "t")]
    assert np.isfinite(rows[:, req]).all()


def test_train_test_split_is_written_and_reproducible(phony, tmp_path):
    """Cases split into train/ and test/ dirs; the seeded split is stable."""

    root, _ = phony
    a = orchestrator.run(root, tmp_path / "a", chunk_size=50, seed=0)
    b = orchestrator.run(root, tmp_path / "b", chunk_size=50, seed=0)

    assert (tmp_path / "a" / "train").is_dir() and (tmp_path / "a" / "test").is_dir()
    assert a["test"], "expected at least one held-out test case"
    # same seed -> same partition
    assert {p.stem for p in a["train"]} == {p.stem for p in b["train"]}
    assert {p.stem for p in a["test"]} == {p.stem for p in b["test"]}


def test_unmapped_type_contributes_no_rows(phony):
    """A canonical=False type (dlaux) is recognized but adds no canonical rows."""

    from config import match_source
    root, _ = phony
    p = next((root / "obs").glob("dlaux_a*"))
    src = match_source(p.name)
    assert src is not None and src.canonical is False
    rows = [to_canonical(ch, src) for ch in read_file(p, src, chunk_size=50)]
    assert all(b.shape[0] == 0 for b in rows), "unmapped type produced rows"
