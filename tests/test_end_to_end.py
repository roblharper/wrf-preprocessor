"""End-to-end test: folder-per-case phony data in, normalized per-case .npy out."""

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

from config import COLUMN_INDEX
from reader import discover_cases, read_file
from processor import to_canonical
from writer import consolidate_stats, write_case_stats
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


# --- folder-per-case discovery -----------------------------------------------
def test_case_ids_come_from_folder_names(phony):
    root, manifest = phony
    cases = discover_cases(root)
    assert sorted(c.name for c in cases) == manifest["case_ids"]


def test_every_data_file_in_a_folder_is_kept(phony):
    """Every recognized non-HRRR file in a folder is a data file (no filtering)."""
    root, manifest = phony
    cases = {c.name: c for c in discover_cases(root)}
    for c in cases.values():
        matches = {src.match for _, src in c.data_files}
        for t in manifest["data_types"]:
            assert t in matches, f"{c.name}: {t} not discovered as data"
        assert c.hrrr is not None and c.hrrr[1].match == "hrrr"


def test_duplicate_case_id_errors():
    from reader import Case
    from config import match_source
    src = match_source("wrfout_d01.nc")
    dup = [Case("x", None, [(Path("a"), src)]), Case("x", None, [(Path("b"), src)])]
    with pytest.raises(ValueError, match="Duplicate case id"):
        orchestrator._dedupe_check(dup)


def test_folder_with_no_data_files_is_skipped(phony, tmp_path, caplog):
    root, _ = phony
    (root / "empty_case").mkdir()
    cases = discover_cases(root)
    assert "empty_case" not in {c.name for c in cases}


# --- pipeline output ----------------------------------------------------------
def test_pipeline_writes_split_cases(phony, tmp_path):
    root, manifest = phony
    out = tmp_path / "out"
    written = orchestrator.run(root, out, chunk_size=50)

    all_stems = {p.stem for p in written["train"] + written["test"]}
    assert all_stems == set(manifest["case_ids"])
    assert (out / "metadata.json").exists()
    assert len(written["test"]) >= 1  # at least one held out
    assert not ({p.stem for p in written["train"]} & {p.stem for p in written["test"]})


def test_hrrr_time_recorded_per_case(phony, tmp_path):
    root, manifest = phony
    out = tmp_path / "out"
    orchestrator.run(root, out, chunk_size=50)

    meta = json.loads((out / "metadata.json").read_text())
    recorded = {}
    for group in meta["cases"].values():
        for cid, info in group.items():
            recorded[cid] = info["hrrr_time"]
    for cid, t in manifest["hrrr_times"].items():
        assert recorded[cid] == pytest.approx(t)


def test_stats_consolidate_to_a_correct_global_recipe(phony, tmp_path):
    """The merged min/max equals a direct min/max over every case's raw rows."""
    root, _ = phony
    work = tmp_path / "work"; work.mkdir()

    from config import CANONICAL_COLUMNS
    from processor import RunningMinMax
    ids, all_rows = [], []
    for case in discover_cases(root):
        stats = RunningMinMax(len(CANONICAL_COLUMNS))
        blocks = []
        for path, src in case.data_files:
            for ch in read_file(path, src, chunk_size=50):
                b = to_canonical(ch, src)
                if b.size:
                    stats.update(b); blocks.append(b)
        rows = np.vstack(blocks)
        np.save(work / f"{case.name}.npy", rows)
        write_case_stats(work, case.name, *stats.result())
        ids.append(case.name); all_rows.append(rows)

    gmin, gmax = consolidate_stats(work, ids)
    stacked = np.vstack(all_rows)
    ref_min = np.nanmin(np.where(np.isnan(stacked), np.inf, stacked), axis=0)
    ref_max = np.nanmax(np.where(np.isnan(stacked), -np.inf, stacked), axis=0)
    finite = np.isfinite(ref_min)
    assert np.allclose(gmin[finite], ref_min[finite])
    assert np.allclose(gmax[finite], ref_max[finite])


def test_output_is_normalized_with_recipe_in_metadata(phony, tmp_path):
    root, _ = phony
    out = tmp_path / "out"
    orchestrator.run(root, out, chunk_size=50)

    meta = json.loads((out / "metadata.json").read_text())
    norm = meta["normalization"]
    offset, scale = norm["offset"], norm["scale"]
    assert len(scale) == len(COLUMN_INDEX) and len(offset) == len(COLUMN_INDEX)
    assert norm["method"] == "minmax_01"

    # minmax_01 maps physical columns into [0, 1]; the source tag is left unscaled.
    phys = [i for c, i in COLUMN_INDEX.items() if c != "source"]
    for f in out.rglob("*.npy"):
        d = np.load(f)[:, phys]
        finite = d[np.isfinite(d)]
        assert np.all(finite >= -1e-9) and np.all(finite <= 1.0 + 1e-9), \
            f"{f.stem}: values outside [0, 1]"


# --- multi-type registry: discovery, derive hooks, unmapped types ------------
def test_discovery_matches_multiple_instrument_types(phony):
    """Every structural type is discovered by filename within a case folder."""
    root, _ = phony
    cases = discover_cases(root)
    matched = {c.hrrr[1].match for c in cases if c.hrrr}
    for _, src in [f for c in cases for f in c.data_files]:
        matched.add(src.match)
    for expected in ("hrrr", "wrfout", "ecorsfwind", "smos", "twr", "co2flx",
                     "armbeatm", "dlaux"):
        assert expected in matched, f"{expected} not discovered"


def test_derive_hooks_produce_correct_canonical_values(phony):
    """ARM time (base+offset) and smos speed/direction -> u, v."""
    root, _ = phony
    case = root / "case_morning"

    def first_canonical(match_glob):
        from config import match_source
        p = next(case.glob(match_glob))
        src = match_source(p.name)
        for ch in read_file(p, src, chunk_size=50):
            b = to_canonical(ch, src)
            if b.size:
                return b
        return None

    # ecor has coordinates, so it survives the NaN-row drop; check absolute time.
    ecor = first_canonical("*ecorsfwind*")
    assert ecor[0, _T] > 1_000_000_000, "ecor t is not an absolute epoch time"

    # smos has no lat/lon (dropped downstream), so test its wind derivation at
    # the hook level: wspd=5, wdir=270 (from the west) -> u ~ +5, v ~ 0.
    from config import _uv_from_speed_dir
    out = _uv_from_speed_dir({"wspd": np.array([5.0]), "wdir": np.array([270.0])})
    assert abs(out["u"][0] - 5.0) < 0.5 and abs(out["v"][0]) < 0.5


def test_source_tag_written_per_category(phony, tmp_path):
    """Each output row carries its source category code (sim/sensor).

    HRRR is the case tag, not a data source, so no HRRR rows appear.
    """
    from config import SRC_SIM, SRC_SENSOR
    root, _ = phony
    out = tmp_path / "out"
    orchestrator.run(root, out, chunk_size=50)

    src_col = COLUMN_INDEX["source"]
    all_tags = set()
    for f in out.rglob("*.npy"):
        all_tags |= set(np.load(f)[:, src_col].astype(int).tolist())
    assert all_tags == {SRC_SIM, SRC_SENSOR}


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
    # normalization is affine, so finiteness is preserved: sim has theta/p',
    # sensor rows carry NaN there.
    assert sim.size and np.isfinite(sim[:, [th, pp]]).all()
    assert sensor.size and np.isnan(sensor[:, [th, pp]]).all()
    req = [COLUMN_INDEX[c] for c in ("x", "y", "z", "t")]
    assert np.isfinite(rows[:, req]).all()


def test_train_test_split_is_written_and_reproducible(phony, tmp_path):
    """Cases split into train/ and test/ dirs; the seeded split is stable."""
    root, _ = phony
    a = orchestrator.run(root, tmp_path / "a", chunk_size=50, seed=0)
    b = orchestrator.run(root, tmp_path / "b", chunk_size=50, seed=0)

    assert (tmp_path / "a" / "train").is_dir() and (tmp_path / "a" / "test").is_dir()
    assert a["test"], "expected at least one held-out test case"
    assert {p.stem for p in a["train"]} == {p.stem for p in b["train"]}
    assert {p.stem for p in a["test"]} == {p.stem for p in b["test"]}


def test_unmapped_type_contributes_no_rows(phony):
    """A canonical=False type (dlaux) is recognized but adds no canonical rows."""
    from config import match_source
    root, _ = phony
    p = next((root / "case_morning").glob("*dlaux*"))
    src = match_source(p.name)
    assert src is not None and src.canonical is False
    rows = [to_canonical(ch, src) for ch in read_file(p, src, chunk_size=50)]
    assert all(b.shape[0] == 0 for b in rows), "unmapped type produced rows"


def test_staggered_winds_collapse_to_mass_grid(phony):
    """Real wrfout has staggered U/V/W; destaggering must not blow up the grid.

    Regression for the 81 TiB outer-product bug: every column must flatten to the
    same mass-grid row count, and Times must parse to an absolute epoch.
    """
    from config import match_source
    root, _ = phony
    p = next((root / "case_morning").glob("wrfout_*"))
    src = match_source(p.name)
    chunk = next(read_file(p, src, chunk_size=50))

    lengths = {len(v) for v in chunk.values()}
    assert len(lengths) == 1, f"columns disagree on row count: {lengths}"
    for name in ("u", "v", "w", "theta", "p_prime"):
        assert name in chunk, f"{name} missing after destagger"
    assert float(chunk["t"][0]) > 1_000_000_000, "Times not parsed to epoch"
