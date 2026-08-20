"""Thin orchestrator: reader -> sync -> processor -> writer.

Wires the stages and nothing more; the pipeline reads off the code:

    for each source folder:
        reader.read_file -> processor.assemble     (raw NetCDF -> canonical rows)
    sync.build_snapshots / attach_data             (bundle each HRRR snapshot with
                                                     its co-located, co-temporal
                                                     LES/sensor rows; drop the rest)
    processor.fit_normalizer  (GLOBAL)             (one recipe over all synced rows)
    writer.write_cases                             (one .npy per snapshot + metadata)

Streaming is preserved through the reader; assembled blocks are consolidated only
at the sync and global-normalization steps, which by design need all rows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from config import ANCHOR_SOURCE, source_for
from reader import iter_source_files, read_file
from processor import assemble, fit_normalizer, MaxNormalizer, Normalizer
from sync import build_snapshots, attach_data
from writer import write_cases


def discover_sources(input_root: Path) -> list[str]:
    """Return the source keys present as subfolders under ``input_root``."""

    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root does not exist: {input_root}")
    return sorted(p.name for p in input_root.iterdir() if p.is_dir())


def run(
    input_root: Path,
    out_dir: Path,
    *,
    chunk_size: int = 1,
    normalizer: Normalizer | None = None,
    verbose: bool = True,
) -> list[Path]:
    """Run the full pipeline and return the written snapshot files."""

    normalizer = normalizer if normalizer is not None else MaxNormalizer()

    # --- read + assemble, split into anchor (HRRR) vs data (LES/sensor) -------
    anchor_blocks: list[np.ndarray] = []
    data_blocks: list[np.ndarray] = []
    for key in discover_sources(input_root):
        try:
            source = source_for(key)
        except KeyError as exc:
            if verbose:
                print(f"  skip: {exc}")
            continue
        target = anchor_blocks if key == ANCHOR_SOURCE else data_blocks
        for path in iter_source_files(input_root / key):
            for chunk in read_file(path, source, chunk_size=chunk_size):
                block = assemble(chunk)
                if block.size:
                    target.append(block)

    # --- sync: one bundle per HRRR snapshot, LES/sensor matched by time+space -
    snapshots = build_snapshots(anchor_blocks)
    kept, dropped = attach_data(snapshots, data_blocks)
    if verbose:
        print(f"  snapshots: {len(snapshots)}; data rows kept: {kept}, dropped: {dropped}")

    # --- global normalization over all synced rows ---------------------------
    all_rows = [snap.rows() for snap in snapshots.values()]
    fit_normalizer(all_rows, normalizer)

    cases = {
        case_id: normalizer.transform(snap.rows())
        for case_id, snap in snapshots.items()
    }

    return write_cases(out_dir, cases, normalizer.recipe())


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Sync NetCDF sources into one normalized .npy per HRRR snapshot.",
    )
    parser.add_argument("input_root", type=Path,
                        help="Folder with one subfolder per source (hrrr/, simulation/, sensor/).")
    parser.add_argument("out_dir", type=Path,
                        help="Output folder for per-snapshot .npy files + metadata.json.")
    parser.add_argument("--chunk-size", type=int, default=1,
                        help="Chunk size along each source's chunk dimension (default 1).")
    args = parser.parse_args(argv)

    written = run(args.input_root, args.out_dir, chunk_size=args.chunk_size)
    print(f"Wrote {len(written)} snapshot file(s) + metadata.json to {args.out_dir}")
    for p in written:
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
