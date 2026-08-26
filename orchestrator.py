"""Wire the pipeline: reader -> to_canonical -> sync -> normalize -> writer."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from reader import discover_files, read_file
from processor import to_canonical, anchor_points, Normalizer
from sync import build_snapshots, attach_data
from writer import write_cases

log = logging.getLogger(__name__)


def run(
    input_root: Path, out_dir: Path, *, chunk_size: int = 1,
    test_fraction: float = 0.2, seed: int = 0,
) -> dict[str, list[Path]]:
    """Run the full pipeline; return the written train/ and test/ case files."""
    anchor_blocks: list[np.ndarray] = []
    data_blocks: list[np.ndarray] = []
    for path, source in discover_files(input_root):
        for chunk in read_file(path, source, chunk_size=chunk_size):
            # HRRR anchors the case (time + bbox); everything else is data.
            if source.is_anchor:
                pts = anchor_points(chunk, source)
                if pts.size:
                    anchor_blocks.append(pts)
            else:
                block = to_canonical(chunk, source)
                if block.size:
                    data_blocks.append(block)

    snapshots = build_snapshots(anchor_blocks)
    kept, dropped = attach_data(snapshots, data_blocks)
    # A snapshot with no synced rows carries no data; skip it.
    cases_with_data = {cid: s for cid, s in snapshots.items() if s.blocks}
    log.info("%d snapshot(s), %d with data; synced %d row(s), dropped %d",
             len(snapshots), len(cases_with_data), kept, dropped)

    normalizer = Normalizer.fit(s.rows() for s in cases_with_data.values())
    cases = {cid: normalizer.transform(s.rows()) for cid, s in cases_with_data.items()}
    return write_cases(out_dir, cases, normalizer.recipe(),
                       test_fraction=test_fraction, seed=seed)


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
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="-v for progress (INFO), -vv for filtering detail (DEBUG).")
    args = parser.parse_args(argv)

    level = [logging.WARNING, logging.INFO, logging.DEBUG][min(args.verbose, 2)]
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    written = run(args.input_root, args.out_dir, chunk_size=args.chunk_size,
                  test_fraction=args.test_fraction, seed=args.seed)
    print(f"Wrote {len(written['train'])} train + {len(written['test'])} test "
          f"case(s) + metadata.json to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
