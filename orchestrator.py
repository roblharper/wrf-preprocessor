"""Wire the pipeline: reader -> to_canonical -> sync -> normalize -> writer."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from reader import discover_files, read_file
from processor import to_canonical, Normalizer
from sync import build_snapshots, attach_data
from writer import write_cases

log = logging.getLogger(__name__)


def run(input_root: Path, out_dir: Path, *, chunk_size: int = 1) -> list[Path]:
    """Run the full pipeline and return the written snapshot files."""
    anchor_blocks: list[np.ndarray] = []
    data_blocks: list[np.ndarray] = []
    for path, source in discover_files(input_root):
        target = anchor_blocks if source.is_anchor else data_blocks
        for chunk in read_file(path, source, chunk_size=chunk_size):
            block = to_canonical(chunk, source)
            if block.size:
                target.append(block)

    snapshots = build_snapshots(anchor_blocks)
    kept, dropped = attach_data(snapshots, data_blocks)
    log.info("%d snapshot(s); synced %d row(s), dropped %d", len(snapshots), kept, dropped)

    normalizer = Normalizer.fit(snap.rows() for snap in snapshots.values())
    cases = {cid: normalizer.transform(snap.rows()) for cid, snap in snapshots.items()}
    return write_cases(out_dir, cases, normalizer.recipe())


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
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="-v for progress (INFO), -vv for filtering detail (DEBUG).")
    args = parser.parse_args(argv)

    level = [logging.WARNING, logging.INFO, logging.DEBUG][min(args.verbose, 2)]
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    written = run(args.input_root, args.out_dir, chunk_size=args.chunk_size)
    print(f"Wrote {len(written)} snapshot file(s) + metadata.json to {args.out_dir}")
    for p in written:
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
