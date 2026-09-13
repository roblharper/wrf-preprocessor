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
    time_tolerance_s: float = TIME_TOLERANCE_SECONDS, device: str = "cpu",
) -> dict[str, list[Path]]:
    """Run the full pipeline; return the written train/ and test/ case files.

    ``device`` runs the per-block math on 'cpu' (numpy) or a GPU ('cuda'); I/O
    (NetCDF read, .npy write) stays on the CPU. ``progress(phase, done, total)``
    reports read/write progress; ``spill_dir`` sets where pass-1 spills go.
    """
    def _tick(phase, done, total):
        if progress is not None:
            progress(phase, done, total)

    files = discover_files(input_root)

    # anchors are tiny (t,x,y triples); read them all to define the snapshots.
    anchor_blocks = [
        pts for path, src in files if src.is_anchor
        for chunk in read_file(path, src, chunk_size=chunk_size)
        for pts in (anchor_points(chunk, src),) if pts.size
    ]
    snapshots = build_snapshots(anchor_blocks)

    with tempfile.TemporaryDirectory(prefix="preproc_spill_", dir=spill_dir) as spill:
        stats = RunningMinMax(len(CANONICAL_COLUMNS))
        spills = _SpillSet(Path(spill), snapshots)
        index = SnapshotIndex(snapshots, tolerance_s=time_tolerance_s)

        # pass 1: read data files once, route rows to per-snapshot spills + stats.
        kept = dropped = 0
        data_files = [(p, s) for p, s in files if not s.is_anchor]
        _tick("read", 0, len(data_files))   # show the phase started before file 1
        xp = array_module(device)
        for i, (path, src) in enumerate(data_files, 1):
            for chunk in read_file(path, src, chunk_size=chunk_size, device=device):
                block = to_canonical(chunk, src, device)
                if block.shape[0] == 0:
                    continue
                k, d = spills.route(block, index, stats, xp=xp)
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
            spills.discard(cid)   # free the spill immediately, cap peak disk
            del rows
            _tick("write", i, len(ids))
        return writer.finalize()


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
