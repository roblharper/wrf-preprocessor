"""Wire the pipeline: reader -> to_canonical -> sync -> normalize -> writer.

Streams so peak memory is one case, not the whole dataset:
  pass 1  read every file once, spill each snapshot's rows to a temp .npy,
          accumulating global min/max on the fly (RunningMinMax)
  pass 2  reload one spill at a time, normalize, write the final case, free it
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np

from config import CANONICAL_COLUMNS, TIME_TOLERANCE_SECONDS
from reader import discover_files, read_file
from processor import to_canonical, anchor_points, Normalizer, RunningMinMax
from sync import build_snapshots, match_snapshots, SnapshotIndex
from writer import CaseWriter
from device import array_module, to_numpy

log = logging.getLogger(__name__)


def run(
    input_root: Path, out_dir: Path, *, chunk_size: int = 1,
    test_fraction: float = 0.2, seed: int = 0, progress=None, spill_dir=None,
    time_tolerance_s: float = TIME_TOLERANCE_SECONDS,
    device: str = "cpu", batch_size: int = 32,
) -> dict[str, list[Path]]:
    """Run the full pipeline; return the written train/ and test/ case files.

    CPU ('cpu') streams one case at a time to disk (bounded RAM). A GPU device
    ('cuda') runs the block math there and holds ``batch_size`` cases resident,
    so the user caps peak device memory. I/O (read, write) stays on the CPU.
    """
    def _tick(phase, done, total):
        if progress is not None:
            progress(phase, done, total)

    files = discover_files(input_root)
    anchor_blocks = [
        pts for path, src in files if src.is_anchor
        for chunk in read_file(path, src, chunk_size=chunk_size)
        for pts in (anchor_points(chunk, src),) if pts.size
    ]
    snapshots = build_snapshots(anchor_blocks)
    index = SnapshotIndex(snapshots, tolerance_s=time_tolerance_s)
    data_files = [(p, s) for p, s in files if not s.is_anchor]

    if device != "cpu":
        return _run_resident_batched(
            data_files, snapshots, index, out_dir, chunk_size=chunk_size,
            test_fraction=test_fraction, seed=seed, tick=_tick,
            time_tolerance_s=time_tolerance_s, device=device, batch_size=batch_size)

    with tempfile.TemporaryDirectory(prefix="preproc_spill_", dir=spill_dir) as spill:
        stats = RunningMinMax(len(CANONICAL_COLUMNS))
        spills = _SpillSet(Path(spill), snapshots)

        # pass 1: read data files once, route rows to per-snapshot spills + stats.
        kept = dropped = 0
        _tick("read", 0, len(data_files))
        for i, (path, src) in enumerate(data_files, 1):
            for chunk in read_file(path, src, chunk_size=chunk_size):
                block = to_canonical(chunk, src)
                if block.shape[0] == 0:
                    continue
                k, d = spills.route(block, index, stats)
                kept += k; dropped += d
            _tick("read", i, len(data_files))

        log.info("%d snapshot(s), %d with data; synced %d row(s), dropped %d",
                 len(snapshots), spills.n_nonempty, kept, dropped)
        normalizer = Normalizer.from_stats(*stats.result(), method="minmax_01")

        # pass 2: reload one case at a time, normalize, write, free.
        ids = spills.nonempty_ids()
        writer = CaseWriter(out_dir, normalizer.recipe(), ids,
                            test_fraction=test_fraction, seed=seed,
                            time_tolerance_s=time_tolerance_s)
        for i, cid in enumerate(ids, 1):
            rows = spills.load(cid)
            writer.add(cid, normalizer.transform(rows))
            spills.discard(cid)
            del rows
            _tick("write", i, len(ids))
        return writer.finalize()


def _canonical_matched(files, index, device, chunk_size, tolerance_s):
    """Yield (cid, device_rows) for every matched case across ``files``. One file
    is one snapshot here (one case), so no cross-file concatenation is needed."""
    xp = array_module(device)
    for path, src in files:
        for chunk in read_file(path, src, chunk_size=chunk_size, device=device):
            block = to_canonical(chunk, src, device)
            if block.shape[0] == 0:
                continue
            kept, _ = match_snapshots(block, index, tolerance_s=tolerance_s, xp=xp)
            for cid, rows in kept.items():
                yield cid, rows


def _run_resident_batched(data_files, snapshots, index, out_dir, *, chunk_size,
                          test_fraction, seed, tick, time_tolerance_s, device,
                          batch_size):
    """GPU path: process ``batch_size`` cases resident at a time so peak device
    memory is one batch (user-capped). Two passes over the files: global stats,
    then normalize + write. No disk spill."""
    batches = [data_files[i:i + batch_size] for i in range(0, len(data_files), batch_size)]

    # pass 1: global min/max + case ids, one batch resident at a time then freed.
    stats = RunningMinMax(len(CANONICAL_COLUMNS))
    ids: list[str] = []
    tick("read", 0, len(data_files))
    done = 0
    for batch in batches:
        for cid, rows in _canonical_matched(batch, index, device, chunk_size, time_tolerance_s):
            stats.update(to_numpy(rows))
            if cid not in ids:
                ids.append(cid)
        done += len(batch)
        tick("read", done, len(data_files))
    normalizer = Normalizer.from_stats(*stats.result(), method="minmax_01")

    writer = CaseWriter(out_dir, normalizer.recipe(), ids,
                        test_fraction=test_fraction, seed=seed,
                        time_tolerance_s=time_tolerance_s)

    # pass 2: one batch resident at a time -> normalize -> write -> free.
    written = 0
    for batch in batches:
        cases: dict[str, list] = {}
        for cid, rows in _canonical_matched(batch, index, device, chunk_size, time_tolerance_s):
            cases.setdefault(cid, []).append(rows)
        for cid, parts in cases.items():
            rows = parts[0] if len(parts) == 1 else _cat(parts, device)
            writer.add(cid, normalizer.transform(to_numpy(rows)))
            written += 1
            tick("write", written, len(ids))
        cases.clear()   # free this batch before the next
    return writer.finalize()


def _cat(parts, device):
    """Concatenate a case's row tensors (torch on GPU, numpy on CPU)."""
    xp = array_module(device)
    return xp.cat(parts) if hasattr(xp, "cat") else xp.concatenate(parts)


class _SpillSet:
    """Per-snapshot append-only row spills on disk, one file per case id."""

    def __init__(self, root: Path, snapshots) -> None:
        self._root = root
        self._counts = {cid: 0 for cid in snapshots}

    def route(self, block, snapshots, stats: RunningMinMax, xp=np) -> tuple[int, int]:
        """Match block to snapshots on ``xp``, then spill each case's rows as host
        numpy (stats + disk are CPU)."""
        kept, matched = match_snapshots(block, snapshots, xp=xp)  # {cid: rows}, mask
        for cid, rows in kept.items():
            rows = to_numpy(rows)              # device -> host at the spill boundary
            stats.update(rows)
            with open(self._root / f"{cid}.npy.part", "ab") as fh:
                fh.write(np.ascontiguousarray(rows).tobytes())
            self._counts[cid] += len(rows)
        n_kept = int(matched.sum())
        return n_kept, int(block.shape[0]) - n_kept

    @property
    def n_nonempty(self) -> int:
        return sum(1 for n in self._counts.values() if n)

    def nonempty_ids(self):
        return [cid for cid, n in self._counts.items() if n]

    def load(self, cid: str) -> np.ndarray:
        raw = np.fromfile(self._root / f"{cid}.npy.part", dtype=np.float64)
        return raw.reshape(-1, len(CANONICAL_COLUMNS))

    def discard(self, cid: str) -> None:
        (self._root / f"{cid}.npy.part").unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Sync NetCDF sources into one normalized .npy per HRRR snapshot.",
    )
    parser.add_argument("input_root", type=Path,
                        help="Folder tree of NetCDF files (matched by filename).")
    parser.add_argument("out_dir", type=Path,
                        help="Output folder for per-snapshot .npy files + metadata.json.")
    parser.add_argument("--chunk-size", type=int, default=1,
                        help="Chunk size along each source's chunk dimension (default 1).")
    parser.add_argument("--test-fraction", type=float, default=0.2,
                        help="Fraction of cases held out for testing (default 0.2).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for the reproducible train/test split.")
    parser.add_argument("--device", default="cpu",
                        help="Compute device for block math: 'cpu' or 'cuda' (default cpu).")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="-v for progress (INFO), -vv for filtering detail (DEBUG).")
    args = parser.parse_args(argv)

    level = [logging.WARNING, logging.INFO, logging.DEBUG][min(args.verbose, 2)]
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    written = run(args.input_root, args.out_dir, chunk_size=args.chunk_size,
                  test_fraction=args.test_fraction, seed=args.seed, device=args.device)
    print(f"Wrote {len(written['train'])} train + {len(written['test'])} test "
          f"case(s) + metadata.json to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
