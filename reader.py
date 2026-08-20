"""Generic, streaming NetCDF reader.

One reader for every source. It does not know what "HRRR" or "sensor" means; it
is handed a :class:`~config.SourceConfig` and streams whatever that config
declares. Source differences live in config, not here.

Memory safety: the file is read in blocks along ``chunk_dim`` (never loaded
whole). Each yielded chunk is a small dict of flat 1-D arrays, one per canonical
column the source supplies, already broadcast to a common row layout. This is
the unit the processor consumes.

Parallelisation is designed-for but not built: chunks are independent, so a
future version can fan reads out per file or per chunk without changing this
interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import netCDF4

from config import SourceConfig


#: A read chunk: canonical column name -> flat 1-D array of equal length.
Chunk = dict[str, np.ndarray]


def iter_source_files(source_dir: Path) -> list[Path]:
    """Return the NetCDF files under a source folder, sorted.

    Accepts the ARM/HRRR extensions seen on disk (.nc, .cdf) and their in-flight
    transfer suffixes (.cdf.v1) are ignored -- only complete files are read.
    """

    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source folder does not exist: {source_dir}")

    files = sorted(
        p for p in source_dir.iterdir()
        if p.is_file() and p.suffix in (".nc", ".cdf")
    )
    if not files:
        raise FileNotFoundError(f"No .nc/.cdf files in {source_dir}")
    return files


def read_file(path: Path, source: SourceConfig, *, chunk_size: int = 1) -> Iterator[Chunk]:
    """Stream one NetCDF file as canonical-column chunks.

    Iterates over ``source.chunk_dim`` in blocks of ``chunk_size`` so the whole
    file is never resident. For each block, every mapped source variable is
    sliced, then flattened and broadcast so all columns share one row axis.

    Coordinate variables that do not span ``chunk_dim`` (e.g. a static lat/lon
    grid, or scalar station coordinates) are broadcast across the block. Columns
    the source does not map are simply absent from the chunk; the processor
    decides how to fill them.
    """

    ds = netCDF4.Dataset(path)
    try:
        if source.chunk_dim not in ds.dimensions:
            raise KeyError(
                f"{path.name}: chunk_dim '{source.chunk_dim}' not a dimension "
                f"(has: {', '.join(ds.dimensions)})."
            )
        n = len(ds.dimensions[source.chunk_dim])

        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            yield _read_block(ds, source, start, stop)
    finally:
        ds.close()


def _read_block(
    ds: "netCDF4.Dataset", source: SourceConfig, start: int, stop: int
) -> Chunk:
    """Slice ``[start:stop)`` along chunk_dim and flatten to aligned columns.

    Different variables span different dimensions (a 4-D grid, a 2-D lat/lon
    grid, a 1-D per-time coordinate, a scalar). They describe the same physical
    points, so each is expanded onto a common set of grid axes -- respecting
    *which* axes it actually spans -- then flattened. This is why we cannot use
    plain trailing-aligned broadcasting: a per-time 1-D column must land on the
    time axis, not the last axis.
    """

    chunk_dim = source.chunk_dim
    sliced: dict[str, tuple[np.ndarray, tuple[str, ...]]] = {}
    for column, varname in source.column_map.items():
        if varname not in ds.variables:
            # Declared in config but not present in this file: skip, let the
            # processor treat the column as missing.
            continue
        var = ds.variables[varname]
        arr, dims = _slice_along(var, chunk_dim, start, stop)
        sliced[column] = (arr, dims)

    return _broadcast_to_rows(sliced)


def _slice_along(
    var: "netCDF4.Variable", chunk_dim: str, start: int, stop: int
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Read a variable (slicing chunk_dim); return its array and dimension names."""

    if chunk_dim in var.dimensions:
        axis = var.dimensions.index(chunk_dim)
        index: list[object] = [slice(None)] * var.ndim
        index[axis] = slice(start, stop)
        return np.asarray(var[tuple(index)]), tuple(var.dimensions)
    # No chunk_dim: a static/scalar field (e.g. lat/lon grid, station alt).
    return np.asarray(var[...]), tuple(var.dimensions)


def _broadcast_to_rows(
    sliced: dict[str, tuple[np.ndarray, tuple[str, ...]]]
) -> Chunk:
    """Expand every column onto a common grid by dimension name, then flatten.

    The common grid is the ordered union of all columns' dimensions (a variable
    with the most dimensions defines the axis order). Each column is placed onto
    the axes it owns and broadcast across the rest, so a per-time coordinate
    tiles correctly across the spatial axes rather than trailing-aligning.
    """

    if not sliced:
        return {}

    # Build the common axis order: start from the widest variable's dims, then
    # append any dims other variables introduce.
    axis_order: list[str] = []
    for _, dims in sorted(sliced.values(), key=lambda ad: -len(ad[1])):
        for d in dims:
            if d not in axis_order:
                axis_order.append(d)

    target_shape = _target_shape(sliced, axis_order)

    rows: Chunk = {}
    for column, (arr, dims) in sliced.items():
        expanded = _place_on_axes(arr, dims, axis_order)
        rows[column] = np.broadcast_to(expanded, target_shape).reshape(-1)
    return rows


def _target_shape(
    sliced: dict[str, tuple[np.ndarray, tuple[str, ...]]], axis_order: list[str]
) -> tuple[int, ...]:
    """Length of each axis in axis_order, taken from whichever column spans it."""

    length: dict[str, int] = {}
    for arr, dims in sliced.values():
        for d, n in zip(dims, arr.shape):
            length[d] = n
    return tuple(length.get(d, 1) for d in axis_order)


def _place_on_axes(
    arr: np.ndarray, dims: tuple[str, ...], axis_order: list[str]
) -> np.ndarray:
    """Reshape ``arr`` so its dimensions sit on their axes in ``axis_order``.

    Axes the variable does not span become size-1 (to be broadcast). A scalar
    (no dims) becomes all size-1.
    """

    shape = [arr.shape[dims.index(d)] if d in dims else 1 for d in axis_order]
    return arr.reshape(shape)
