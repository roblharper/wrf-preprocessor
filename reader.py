"""Generic streaming NetCDF reader, driven by a SourceType record."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import numpy as np
import netCDF4

from config import SourceType, match_source

log = logging.getLogger(__name__)

#: name -> flat 1-D array (canonical columns and raw derive-inputs).
Chunk = dict[str, np.ndarray]


def discover_files(input_root: Path) -> list[tuple[Path, SourceType]]:
    """Walk the tree, match each .nc/.cdf file to a SourceType by filename.

    Any folder layout works. Unmatched files are reported and skipped.
    """
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root does not exist: {input_root}")

    matched: list[tuple[Path, SourceType]] = []
    unmatched: list[Path] = []
    for p in sorted(input_root.rglob("*")):
        if not (p.is_file() and p.suffix in (".nc", ".cdf")):
            continue
        source = match_source(p.name)
        if source is not None:
            matched.append((p, source))
            log.debug("matched %s -> %s", p.name, source.name)
        else:
            unmatched.append(p)

    if unmatched:
        names = ", ".join(sorted({u.name for u in unmatched})[:6])
        log.info("%d file(s) matched no type, skipped: %s", len(unmatched), names)
    if not matched:
        raise FileNotFoundError(f"No recognized NetCDF files under {input_root}.")
    return matched


def read_file(path: Path, source: SourceType, *, chunk_size: int = 1) -> Iterator[Chunk]:
    """Stream a file in blocks along ``chunk_dim`` (never loaded whole)."""
    ds = netCDF4.Dataset(path)
    try:
        if source.chunk_dim not in ds.dimensions:
            raise KeyError(
                f"{path.name}: chunk_dim '{source.chunk_dim}' not a dimension "
                f"(has: {', '.join(ds.dimensions)})."
            )
        n = len(ds.dimensions[source.chunk_dim])
        for start in range(0, n, chunk_size):
            yield _read_block(ds, source, start, min(start + chunk_size, n))
    finally:
        ds.close()


def _read_block(ds: "netCDF4.Dataset", source: SourceType, start: int, stop: int) -> Chunk:
    """Read the column_map + derive_inputs variables for [start, stop)."""
    wanted = list(source.column_map.items()) + [(v, v) for v in source.derive_inputs]
    sliced: dict[str, tuple[np.ndarray, tuple[str, ...]]] = {}
    for out_key, varname in wanted:
        if varname in ds.variables:
            sliced[out_key] = _slice_along(ds.variables[varname], source.chunk_dim, start, stop)
    return _broadcast_to_rows(sliced)


def _slice_along(
    var: "netCDF4.Variable", chunk_dim: str, start: int, stop: int
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Read a variable, slicing chunk_dim if present; return array + dim names."""
    if chunk_dim in var.dimensions:
        index: list[object] = [slice(None)] * var.ndim
        index[var.dimensions.index(chunk_dim)] = slice(start, stop)
        return np.asarray(var[tuple(index)]), tuple(var.dimensions)
    return np.asarray(var[...]), tuple(var.dimensions)


def _broadcast_to_rows(sliced: dict[str, tuple[np.ndarray, tuple[str, ...]]]) -> Chunk:
    """Expand every column onto a common grid by dimension name, then flatten.

    Broadcasting by name (not trailing-aligned) so a per-time coordinate tiles
    across the spatial axes rather than landing on the wrong axis.
    """
    if not sliced:
        return {}

    axis_order: list[str] = []
    for _, dims in sorted(sliced.values(), key=lambda ad: -len(ad[1])):
        for d in dims:
            if d not in axis_order:
                axis_order.append(d)

    target_shape = _target_shape(sliced, axis_order)
    return {
        column: np.broadcast_to(_place_on_axes(arr, dims, axis_order), target_shape).reshape(-1)
        for column, (arr, dims) in sliced.items()
    }


def _target_shape(
    sliced: dict[str, tuple[np.ndarray, tuple[str, ...]]], axis_order: list[str]
) -> tuple[int, ...]:
    length: dict[str, int] = {}
    for arr, dims in sliced.values():
        for d, n in zip(dims, arr.shape):
            length[d] = n
    return tuple(length.get(d, 1) for d in axis_order)


def _place_on_axes(arr: np.ndarray, dims: tuple[str, ...], axis_order: list[str]) -> np.ndarray:
    """Reshape so each dimension sits on its axis in axis_order; others size-1."""
    shape = [arr.shape[dims.index(d)] if d in dims else 1 for d in axis_order]
    return arr.reshape(shape)
