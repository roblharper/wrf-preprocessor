"""Generate phony ARM-named NetCDFs (one per structural type) for testing.

Includes >=2 of each core source at a shared timestamp, one off-timestamp, and
deliberate off-time / off-location rows the pipeline must drop.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import netCDF4

warnings.filterwarnings(
    "ignore",
    message="Setting the shape on a NumPy array has been deprecated",
    category=DeprecationWarning,
)

SYNC_TIME = 1_720_580_400.0
REGION_LON = (-97.60, -97.35)
REGION_LAT = (36.50, 36.70)
REGION_Z = (0.0, 1000.0)
_DAY0 = SYNC_TIME - (SYNC_TIME % 86400.0)      # midnight base for ARM offsets


# --- anchor + interior (gridded snapshots) ----------------------------------
def _write_hrrr(path: Path, *, valid_time: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    ny = nx = 4; nlev = 3
    ds = netCDF4.Dataset(path, "w")
    ds.createDimension("time", 1); ds.createDimension("hybrid_level", nlev)
    ds.createDimension("y", ny); ds.createDimension("x", nx)
    lon = np.linspace(*REGION_LON, nx); lat = np.linspace(*REGION_LAT, ny)
    lon2d, lat2d = np.meshgrid(lon, lat)
    ds.createVariable("longitude", "f4", ("y", "x"))[:, :] = lon2d
    ds.createVariable("latitude", "f4", ("y", "x"))[:, :] = lat2d
    ds.createVariable("valid_time", "f8", ("time",))[:] = [valid_time]
    shape = (1, nlev, ny, nx)
    for name, base in (("u_wind", 5.0), ("v_wind", 2.0), ("vertical_velocity", 0.1),
                       ("temperature", 290.0), ("pressure", 95000.0),
                       ("geopotential_height", 500.0)):
        ds.createVariable(name, "f4", ("time", "hybrid_level", "y", "x"))[:, :, :, :] = \
            base + rng.normal(0, 0.1, size=shape)
    ds.close()


def _write_les(path: Path, *, valid_time: float, seed: int, lon_c: float, lat_c: float) -> None:
    rng = np.random.default_rng(seed)
    n = 5; nz = 3
    ds = netCDF4.Dataset(path, "w")
    ds.createDimension("time", 1); ds.createDimension("z4", nz)
    ds.createDimension("y0", n); ds.createDimension("x0", n)
    ds.createVariable("x0", "f8", ("x0",))[:] = np.linspace(lon_c - 0.08, lon_c + 0.08, n)
    ds.createVariable("y0", "f8", ("y0",))[:] = np.linspace(lat_c - 0.06, lat_c + 0.06, n)
    ds.createVariable("z4", "f8", ("z4",))[:] = np.linspace(*REGION_Z, nz)
    ds.createVariable("time", "f8", ("time",))[:] = [valid_time]
    shape = (1, nz, n, n)
    for name, base in (("uu", 5.0), ("vv", 2.0), ("ww", 0.1),
                       ("pressure", 95000.0), ("theta", 300.0)):
        ds.createVariable(name, "f4", ("time", "z4", "y0", "x0"))[:, :, :, :] = \
            base + rng.normal(0, 0.1, size=shape)
    ds.close()


# --- ARM observation streams (base_time + time_offset) ----------------------
def _arm_time(ds, times: np.ndarray) -> None:
    """Write ARM-style time: base_time (scalar epoch) + time_offset (s)."""

    ds.createVariable("base_time", "f8", ())[...] = _DAY0
    ds.createVariable("time_offset", "f8", ("time",))[:] = times - _DAY0


def _write_ecor(path: Path, *, times: np.ndarray, seed: int, lon, lat, alt) -> None:
    rng = np.random.default_rng(seed)
    ds = netCDF4.Dataset(path, "w"); ds.createDimension("time", len(times))
    _arm_time(ds, times)
    ds.createVariable("lon", "f8", ())[...] = lon
    ds.createVariable("lat", "f8", ())[...] = lat
    ds.createVariable("alt", "f8", ())[...] = alt
    for name, base in (("wind_u", 5.0), ("wind_v", 2.0), ("wind_w", 0.1)):
        ds.createVariable(name, "f4", ("time",))[:] = base + rng.normal(0, 0.2, len(times))
    ds.close()


def _write_smos(path: Path, *, times: np.ndarray, seed: int) -> None:
    """Soil-met: wind as speed/direction (derive -> u,v); no lat/lon in file."""

    rng = np.random.default_rng(seed)
    ds = netCDF4.Dataset(path, "w"); ds.createDimension("time", len(times))
    _arm_time(ds, times)
    ds.createVariable("wspd", "f4", ("time",))[:] = 5.0 + rng.normal(0, 0.3, len(times))
    ds.createVariable("wdir", "f4", ("time",))[:] = 270.0 + rng.normal(0, 5, len(times))
    ds.createVariable("temp", "f4", ("time",))[:] = 20.0 + rng.normal(0, 0.5, len(times))
    ds.close()


def _write_twr(path: Path, *, times: np.ndarray, seed: int, lon, lat, alt) -> None:
    """Tower met: Celsius temperature (derive -> T), lat/lon/alt, no wind."""

    rng = np.random.default_rng(seed)
    ds = netCDF4.Dataset(path, "w"); ds.createDimension("time", len(times))
    _arm_time(ds, times)
    ds.createVariable("lon", "f8", ())[...] = lon
    ds.createVariable("lat", "f8", ())[...] = lat
    ds.createVariable("alt", "f8", ())[...] = alt
    ds.createVariable("temp", "f4", ("time",))[:] = 18.0 + rng.normal(0, 0.4, len(times))
    ds.createVariable("rh", "f4", ("time",))[:] = 60.0 + rng.normal(0, 2, len(times))
    ds.close()


def _write_co2flx(path: Path, *, times: np.ndarray, seed: int) -> None:
    """Flux + met: mostly fluxes; a little canonical state (bar_pres -> P)."""

    rng = np.random.default_rng(seed)
    ds = netCDF4.Dataset(path, "w"); ds.createDimension("time", len(times))
    _arm_time(ds, times)
    ds.createVariable("bar_pres", "f4", ("time",))[:] = 95.0 + rng.normal(0, 0.1, len(times))
    ds.createVariable("h", "f4", ("time",))[:] = rng.normal(100, 10, len(times))     # sensible heat
    ds.createVariable("le", "f4", ("time",))[:] = rng.normal(150, 15, len(times))    # latent heat
    ds.close()


def _write_armbeatm(path: Path, *, times: np.ndarray, seed: int) -> None:
    """Best-estimate profiles: vertical dims + surface wind (mapped)."""

    rng = np.random.default_rng(seed)
    nheight = 6
    ds = netCDF4.Dataset(path, "w")
    ds.createDimension("time", len(times)); ds.createDimension("height", nheight)
    _arm_time(ds, times)
    ds.createVariable("height", "f8", ("height",))[:] = np.linspace(10, 2000, nheight)
    ds.createVariable("u_wind_sfc", "f4", ("time",))[:] = 5.0 + rng.normal(0, 0.3, len(times))
    ds.createVariable("v_wind_sfc", "f4", ("time",))[:] = 2.0 + rng.normal(0, 0.3, len(times))
    ds.close()


def _write_dlaux(path: Path, *, times: np.ndarray, seed: int) -> None:
    """Doppler lidar housekeeping: no canonical fields (registered, unmapped)."""

    rng = np.random.default_rng(seed)
    ds = netCDF4.Dataset(path, "w"); ds.createDimension("time", len(times))
    _arm_time(ds, times)
    ds.createVariable("battery_voltage", "f4", ("time",))[:] = 12.0 + rng.normal(0, 0.1, len(times))
    ds.createVariable("pitch", "f4", ("time",))[:] = rng.normal(0, 0.5, len(times))
    ds.close()


def generate(root: Path, *, seed: int = 0) -> dict:
    """Build the phony input tree under ``root``; return a manifest for tests."""

    for sub in ("hrrr", "sim", "obs"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    off_time = SYNC_TIME + 7 * 3600.0
    lon_c, lat_c = float(np.mean(REGION_LON)), float(np.mean(REGION_LAT))
    in_window = SYNC_TIME + np.linspace(-600, 600, 20)          # within +/-30 min
    out_window = SYNC_TIME + np.array([5 * 3600.0, 6 * 3600.0])  # outside window
    straddle = np.concatenate([in_window, out_window])

    def n(kind, tag): return f"sgp{kind}{tag}C1.b1.20240710.000000"

    # anchor: 2 at SYNC_TIME + 1 off-time
    _write_hrrr(root / "hrrr" / "hrrr_a.nc", valid_time=SYNC_TIME, seed=seed + 1)
    _write_hrrr(root / "hrrr" / "hrrr_b.nc", valid_time=SYNC_TIME, seed=seed + 2)
    _write_hrrr(root / "hrrr" / "hrrr_off.nc", valid_time=off_time, seed=seed + 3)

    # interior LES: 2 at SYNC_TIME + 1 off-time
    _write_les(root / "sim" / "les_a.nc", valid_time=SYNC_TIME, seed=seed + 4, lon_c=lon_c, lat_c=lat_c)
    _write_les(root / "sim" / "les_b.nc", valid_time=SYNC_TIME, seed=seed + 5, lon_c=lon_c, lat_c=lat_c)
    _write_les(root / "sim" / "les_off.nc", valid_time=off_time, seed=seed + 6, lon_c=lon_c, lat_c=lat_c)

    o = root / "obs"
    # ecor sonic wind: 2 in-region stations (straddle window) + off-time + far
    _write_ecor(o / "ecorsfwind_a.nc", times=straddle, seed=seed + 7,
                lon=lon_c - 0.02, lat=lat_c + 0.01, alt=320.0)
    _write_ecor(o / "ecorsfwind_b.nc", times=in_window, seed=seed + 8,
                lon=lon_c + 0.03, lat=lat_c - 0.02, alt=330.0)
    _write_ecor(o / "ecorsfwind_off.nc", times=off_time + (in_window - SYNC_TIME),
                seed=seed + 9, lon=lon_c, lat=lat_c, alt=325.0)
    _write_ecor(o / "ecorsfwind_far.nc", times=in_window, seed=seed + 10,
                lon=lon_c + 5.0, lat=lat_c + 5.0, alt=300.0)     # off-location

    # one file of each remaining ARM type at the sync window (structural coverage)
    _write_smos(o / "smos_a.nc", times=in_window, seed=seed + 11)          # no lat/lon -> no bbox match
    _write_twr(o / "twr25m_a.nc", times=in_window, seed=seed + 12,
               lon=lon_c - 0.01, lat=lat_c + 0.02, alt=316.0)
    _write_co2flx(o / "co2flx4mmet_a.nc", times=in_window, seed=seed + 13)  # no lat/lon
    _write_armbeatm(o / "armbeatm_a.nc", times=in_window, seed=seed + 14)   # no lat/lon
    _write_dlaux(o / "dlaux_a.nc", times=in_window, seed=seed + 15)         # unmapped

    return {
        "sync_time": SYNC_TIME,
        "off_time": off_time,
        "region_lon": REGION_LON,
        "region_lat": REGION_LAT,
        "in_window_times": in_window.tolist(),
        "off_location_file": "ecorsfwind_far",
        "off_time_file": "ecorsfwind_off",
        "unmapped_file": "dlaux_a",
    }


if __name__ == "__main__":
    import sys
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("fixtures/generated")
    manifest = generate(dest)
    print(f"wrote phony data to {dest}")
    for k, v in manifest.items():
        print(f"  {k}: {v}")
