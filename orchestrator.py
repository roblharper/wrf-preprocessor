"""Wire the pipeline: discover cases -> raw per-case rows -> normalize -> writer.

Folder-per-case with a mergeable-stats seam, so the expensive per-case work is
independent and resumable:
  phase 1  per case: read every file, to_canonical, concatenate -> raw <case>.npy
           + <case>.stats.json (per-column min/max). Cases never interact.
  consolidate  merge every stats.json -> one global min/max -> normalize recipe.
  phase 2  per case: re-read the raw .npy, normalize with the global recipe,
           write the final case; train/test split + metadata.
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

import numpy as np

from config import CANONICAL_COLUMNS
from reader import Case, discover_cases, read_file
from processor import to_canonical, Normalizer, RunningMinMax
from sync import read_hrrr_time
from writer import CaseWriter, consolidate_stats, write_case_stats
from device import array_module, to_numpy, step

log = logging.getLogger(__name__)

NORM_METHOD = "minmax_01"


def _meter(phase, done, total):
    """Default progress: a one-line bar per phase on stderr, newline when full."""
    width = 30
    filled = width * done // max(total, 1)
    bar = "#" * filled + "-" * (width - filled)
    end = "\n" if done >= total else "\r"
    print(f"\r{phase:5s} [{bar}] {done}/{total}", end=end, file=sys.stderr, flush=True)


def run(
    input_root: Path, out_dir: Path, *, chunk_size: int = 1,
    test_fraction: float = 0.2, seed: int = 0, progress=None, work_dir=None,
    device: str = "cpu",
) -> dict[str, list[Path]]:
    """Run the full pipeline; return the written train/ and test/ case files.

    Each ``input_root/<case>/`` folder is one case (one HRRR file + all LES/sensor
    files in it). ``device`` selects the block math: 'cpu' (numpy) or a GPU device
    (torch); I/O (read raw .npy, write) stays on the host.
    """
    def _tick(phase, done, total):
        (progress or _meter)(phase, done, total)

    cases = discover_cases(input_root)
    _dedupe_check(cases)

    with tempfile.TemporaryDirectory(prefix="preproc_raw_", dir=work_dir) as work:
        work = Path(work)

        # phase 1: per case, read + assemble raw rows, write raw .npy + stats.
        hrrr_times: dict[str, float] = {}
        _tick("read", 0, len(cases))
        for i, case in enumerate(cases, 1):
            n = _phase1_case(case, work, chunk_size=chunk_size, device=device)
            if case.hrrr is not None:
                hrrr_times[case.name] = read_hrrr_time(*case.hrrr, chunk_size=chunk_size)
            log.info("case %s: %d row(s)", case.name, n)
            _tick("read", i, len(cases))

        ids = [c.name for c in cases if (work / f"{c.name}.npy").exists()]
        if not ids:
            raise RuntimeError("No rows produced by any case; nothing to normalize.")

        # consolidate: merge per-case stats -> one global recipe.
        col_min, col_max = consolidate_stats(work, ids)
        normalizer = Normalizer.from_stats(col_min, col_max, method=NORM_METHOD)

        # phase 2: per case, re-read raw, normalize, write final.
        writer = CaseWriter(out_dir, normalizer.recipe(), ids,
                            test_fraction=test_fraction, seed=seed,
                            hrrr_times=hrrr_times)
        dev_norm = normalizer.on_device(device)   # offset/scale resident on GPU
        _tick("write", 0, len(ids))
        for i, cid in enumerate(ids, 1):
            _phase2_case(cid, work, writer, dev_norm, device)
            _tick("write", i, len(ids))
        return writer.finalize()


def _phase1_case(case: Case, work: Path, *, chunk_size: int, device: str) -> int:
    """Read every data file in a case, assemble raw rows, write raw .npy + stats.

    Returns the row count. Peak memory is one case's rows (plus one read block).
    """
    xp = array_module(device)
    stats = RunningMinMax(len(CANONICAL_COLUMNS))
    blocks: list = []
    for path, src in case.data_files:
        for chunk in read_file(path, src, chunk_size=chunk_size, device=device):
            with step("canonical", device):
                block = to_canonical(chunk, src, device)
            if block.shape[0] == 0:
                continue
            stats.update(block, xp)      # on-device: no per-block host copy
            blocks.append(block)
    if not blocks:
        return 0

    rows = blocks[0] if len(blocks) == 1 else _cat(blocks, device)
    with step("dev_to_host", device):
        host = to_numpy(rows)
    np.save(work / f"{case.name}.npy", host)
    write_case_stats(work, case.name, *stats.result())
    return int(host.shape[0])


def _phase2_case(cid: str, work: Path, writer: "CaseWriter", dev_norm, device) -> None:
    """Re-read one raw case, normalize on device, write the final case, free it."""
    raw = np.load(work / f"{cid}.npy")
    if device == "cpu":
        writer.add(cid, dev_norm.transform(raw))
        return
    from device import to_device
    rows = to_device(raw, device)
    with step("normalize", device):
        rows = dev_norm.transform(rows)
    with step("dev_to_host", device):
        writer.add(cid, to_numpy(rows))


def _dedupe_check(cases: list[Case]) -> None:
    """Case id = folder name; a collision is a user error, never a silent merge."""
    seen: set[str] = set()
    for c in cases:
        if c.name in seen:
            raise ValueError(f"Duplicate case id '{c.name}'; folder names must be unique.")
        seen.add(c.name)


def _cat(parts, device):
    """Concatenate a case's row tensors (torch on GPU, numpy on CPU)."""
    xp = array_module(device)
    return xp.cat(parts) if hasattr(xp, "cat") else xp.concatenate(parts)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess one normalized .npy per case folder (folder-per-case).",
    )
    parser.add_argument("input_root", type=Path,
                        help="Folder of case subfolders: input_root/<case>/ NetCDF files.")
    parser.add_argument("out_dir", type=Path,
                        help="Output folder for per-case .npy files + metadata.json.")
    parser.add_argument("--chunk-size", type=int, default=1,
                        help="Chunk size along each source's chunk dimension (default 1).")
    parser.add_argument("--test-fraction", type=float, default=0.2,
                        help="Fraction of cases held out for testing (default 0.2).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for the reproducible train/test split.")
    parser.add_argument("--device", default="cpu",
                        help="Compute device for block math: 'cpu' or 'cuda' (default cpu).")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="-v for progress (INFO), -vv for detail (DEBUG).")
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
