"""Canonical schema and per-source variable maps for the pre-processor.

This is the single, evolving place that decides:

  1. the canonical output schema the PINN will consume, and
  2. how each raw NetCDF source's variables/coordinates map onto that schema.

Everything downstream (reader, processor, writer) is generic and driven by what
this module declares. Adding or changing a source is a config edit here, not new
code -- the sources are all NetCDF, so their differences are *data*, not
*behaviour*.

NOTE (rough draft): the schema below is a working assumption pending the
advisor's guidance. Density is not stored by any source; the working schema uses
temperature and pressure (``T``, ``P``) and leaves density derivation to a later
processing step. Coordinate/time reconciliation between sources is deliberately
left as a seam (see ``SourceConfig.coordinate_note``); it is not solved here.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# --- canonical schema --------------------------------------------------------
#: The canonical output columns, in order. The writer emits rows in exactly this
#: layout; the PINN reads exactly this. Evolving -- one place to change.
CANONICAL_COLUMNS: tuple[str, ...] = (
    "x", "y", "z", "t",          # coordinates
    "u", "v", "w",               # velocity
    "T", "P",                    # thermodynamics (working assumption)
)

#: Column-name -> index, so the sync/normalize stages address x/y/t by meaning
#: rather than a magic number.
COLUMN_INDEX: dict[str, int] = {name: i for i, name in enumerate(CANONICAL_COLUMNS)}

#: The source key that anchors a case. A case is one HRRR snapshot; LES and
#: sensor rows are synced to it (see ``sync``).
ANCHOR_SOURCE = "hrrr"

#: Half-width of the time-match tolerance window, in seconds. LES / sensor rows
#: whose ``t`` is within +/- this of an HRRR snapshot time are associated with
#: that snapshot; rows outside every window are dropped. Sensors are ~10 Hz and
#: will not hit a snapshot's exact second, so an exact match is too strict.
TIME_TOLERANCE_SECONDS = 1800.0   # +/- 30 min (half of hourly HRRR cadence)


@dataclass(frozen=True)
class SourceConfig:
    """How one NetCDF source maps onto the canonical schema.

    Attributes
    ----------
    key:
        Source key. Also the input subfolder name (e.g. ``hrrr/``), so the
        folder-to-source mapping is a lookup, not a guess.
    column_map:
        ``canonical_column -> source_variable_name``. A canonical column absent
        from this map is not supplied by the source (left to the processor to
        fill or mark missing). Coordinate columns may map to coordinate
        variables or dimensions; the processor resolves them.
    chunk_dim:
        The dimension the reader iterates over to stream the file in blocks
        (memory-safe). For gridded sources this is usually the time dimension.
    coordinate_note:
        Free-text marker describing the source's native coordinate/time frame.
        A reminder that reconciling these frames is an unresolved seam, not a
        promise that the reader does it.
    """

    key: str
    column_map: dict[str, str]
    chunk_dim: str
    coordinate_note: str = ""


# --- per-source registry -----------------------------------------------------
# Keyed by source key == input subfolder name. Verified against the on-disk
# files (variable and dimension names read directly, not assumed).

HRRR = SourceConfig(
    key="hrrr",
    column_map={
        # coordinates: HRRR is lat/lon on hybrid pressure levels; these map to
        # the native fields for now. Projecting lat/lon -> a metric (x, y) frame
        # and hybrid levels -> physical height is the reconciliation seam.
        "x": "longitude",
        "y": "latitude",
        "z": "geopotential_height",
        "t": "valid_time",
        "u": "u_wind",
        "v": "v_wind",
        "w": "vertical_velocity",
        "T": "temperature",
        "P": "pressure",
    },
    chunk_dim="time",
    coordinate_note="lat/lon degrees on 21 hybrid pressure levels; 7x7 tile, 24 forecast hours",
)

# Simulation (FastEddy LES) and sensor (ecor sonic wind) entries are declared as
# the sources come online. They follow the same shape; only the column_map and
# chunk_dim change. Left minimal on purpose (rough draft, HRRR is the anchor).
SIMULATION = SourceConfig(
    key="simulation",
    column_map={
        "x": "x0", "y": "y0", "z": "z4", "t": "time",
        "u": "uu", "v": "vv", "w": "ww",
        "P": "pressure",
        # FastEddy stores potential temperature (theta), not T -- converting
        # theta -> T is a processing step, so T is intentionally unmapped here.
    },
    chunk_dim="time",
    coordinate_note="Cartesian metres, 45x45x19 grid, single snapshot; theta not T",
)

SENSOR = SourceConfig(
    key="sensor",
    column_map={
        # single-point stations: lat/lon/alt are scalars, broadcast across every
        # time row by the reader. Same lat/lon -> x/y mapping caveat as HRRR.
        "x": "lon", "y": "lat", "z": "alt",
        "t": "time",
        "u": "wind_u", "v": "wind_v", "w": "wind_w",
        # no T/P on this stream; those columns stay NaN (not supplied).
    },
    chunk_dim="time",
    coordinate_note="single-point sonic wind at ~10 Hz; lat/lon/alt scalars, no T/P",
)


#: Registry the orchestrator looks source keys up in. Add a source = add a line.
SOURCES: dict[str, SourceConfig] = {s.key: s for s in (HRRR, SIMULATION, SENSOR)}


def source_for(key: str) -> SourceConfig:
    """Return the config for a source key, or raise a clear error."""

    try:
        return SOURCES[key]
    except KeyError:
        known = ", ".join(sorted(SOURCES))
        raise KeyError(
            f"No source config for '{key}'. Known source keys: {known}. "
            f"(The input subfolder name must match a source key.)"
        ) from None
