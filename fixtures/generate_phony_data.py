"""Generate phony NetCDF data matching the real source formats, for testing.

Writes an input tree the pre-processor can consume:

    <root>/
        hrrr/        *.nc   (variables/dims as config.HRRR expects)
        simulation/  *.nc   (as config.SIMULATION)
        sensor/      *.nc   (as config.SENSOR)

Design (per the test plan):
  * >= 2 files of each source share a common timestamp (the "in-sync" set).
  * 1 file of each source uses a random off-timestamp (tests that off-time data
    is dropped, not misattributed).
  * A small, controllable fraction of LES/sensor rows is deliberately shifted
    off in time and/or space, so the end-to-end test can assert those rows are
    excluded from the synced output.

The variable and dimension names mirror what the real files carry (verified
earlier against the on-disk data), so the generic reader handles them unchanged.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import netCDF4

# netCDF4 1.7.x triggers a NumPy 2.5 DeprecationWarning inside its own C-layer
# array assignment (not from our code); silence that specific noise.
warnings.filterwarnings(
    "ignore",
    message="Setting the shape on a NumPy array has been deprecated",
    category=DeprecationWarning,
)


# A shared reference time (epoch seconds) and a spatial region the "in-sync"
# data all agree on. Kept small so tests are fast.
SYNC_TIME = 1_720_580_400.0          # arbitrary fixed epoch second
REGION_LON = (-97.60, -97.35)        # HRRR bbox in lon
REGION_LAT = (36.50, 36.70)          # HRRR bbox in lat
REGION_Z = (0.0, 1000.0)


def _write_hrrr(path: Path, *, valid_time: float, seed: int) -> None:
    """A small HRRR-format tile: (time, hybrid_level, y, x) grids + lat/lon."""

    rng = np.random.default_rng(seed)
    ny = nx = 4
    nlev = 3
    ds = netCDF4.Dataset(path, "w")
    ds.createDimension("time", 1)
    ds.createDimension("hybrid_level", nlev)
    ds.createDimension("y", ny)
    ds.createDimension("x", nx)

    lon = np.linspace(REGION_LON[0], REGION_LON[1], nx)
    lat = np.linspace(REGION_LAT[0], REGION_LAT[1], ny)
    lon2d, lat2d = np.meshgrid(lon, lat)

    ds.createVariable("longitude", "f4", ("y", "x"))[:, :] = lon2d
    ds.createVariable("latitude", "f4", ("y", "x"))[:, :] = lat2d
    ds.createVariable("valid_time", "f8", ("time",))[:] = [valid_time]

    shape = (1, nlev, ny, nx)
    for name, base in (("u_wind", 5.0), ("v_wind", 2.0), ("vertical_velocity", 0.1),
                       ("temperature", 290.0), ("pressure", 95000.0),
                       ("geopotential_height", 500.0)):
        var = ds.createVariable(name, "f4", ("time", "hybrid_level", "y", "x"))
        var[:, :, :, :] = base + rng.normal(0, 0.1, size=shape)
    ds.close()


def _write_simulation(path: Path, *, valid_time: float, seed: int,
                      lon_center: float, lat_center: float) -> None:
    """A small FastEddy-format LES snapshot: (time, z4, y0, x0)."""

    rng = np.random.default_rng(seed)
    n = 5
    nz = 3
    ds = netCDF4.Dataset(path, "w")
    ds.createDimension("time", 1)
    ds.createDimension("z4", nz)
    ds.createDimension("y0", n)
    ds.createDimension("x0", n)

    # LES x0/y0 are metres in the real files, but for a self-consistent phony
    # test we place them on the same lon/lat frame as HRRR so bbox matching is
    # meaningful. (Real coordinate reconciliation is a separate, later step.)
    x0 = np.linspace(lon_center - 0.08, lon_center + 0.08, n)
    y0 = np.linspace(lat_center - 0.06, lat_center + 0.06, n)
    ds.createVariable("x0", "f8", ("x0",))[:] = x0
    ds.createVariable("y0", "f8", ("y0",))[:] = y0
    ds.createVariable("z4", "f8", ("z4",))[:] = np.linspace(*REGION_Z, nz)
    ds.createVariable("time", "f8", ("time",))[:] = [valid_time]

    shape = (1, nz, n, n)
    for name, base in (("uu", 5.0), ("vv", 2.0), ("ww", 0.1),
                       ("pressure", 95000.0), ("theta", 300.0)):
        ds.createVariable(name, "f4", ("time", "z4", "y0", "x0"))[:, :, :, :] = \
            base + rng.normal(0, 0.1, size=shape)
    ds.close()


def _write_sensor(path: Path, *, times: np.ndarray, seed: int,
                  lon: float, lat: float, alt: float) -> None:
    """A small ecor-format sensor stream: single point, many time samples."""

    rng = np.random.default_rng(seed)
    n = len(times)
    ds = netCDF4.Dataset(path, "w")
    ds.createDimension("time", n)
    ds.createVariable("time", "f8", ("time",))[:] = times
    ds.createVariable("lon", "f8", ())[...] = lon
    ds.createVariable("lat", "f8", ())[...] = lat
    ds.createVariable("alt", "f8", ())[...] = alt
    for name, base in (("wind_u", 5.0), ("wind_v", 2.0), ("wind_w", 0.1)):
        ds.createVariable(name, "f4", ("time",))[:] = base + rng.normal(0, 0.2, n)
    ds.close()


def generate(root: Path, *, off_fraction: float = 0.1, seed: int = 0) -> dict:
    """Build the phony input tree under ``root``. Returns a manifest of what was
    written (times, expected in-sync counts) for tests to assert against.
    """

    rng = np.random.default_rng(seed)
    for sub in ("hrrr", "simulation", "sensor"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    off_time = SYNC_TIME + 7 * 3600.0          # far outside the tolerance window
    lon_c = np.mean(REGION_LON)
    lat_c = np.mean(REGION_LAT)

    # --- HRRR: two snapshots at SYNC_TIME (two tiles) + one off-time ----------
    _write_hrrr(root / "hrrr" / "hrrr_a.nc", valid_time=SYNC_TIME, seed=seed + 1)
    _write_hrrr(root / "hrrr" / "hrrr_b.nc", valid_time=SYNC_TIME, seed=seed + 2)
    _write_hrrr(root / "hrrr" / "hrrr_off.nc", valid_time=off_time, seed=seed + 3)

    # --- simulation: two at SYNC_TIME + one off-time --------------------------
    _write_simulation(root / "simulation" / "les_a.nc", valid_time=SYNC_TIME,
                      seed=seed + 4, lon_center=lon_c, lat_center=lat_c)
    _write_simulation(root / "simulation" / "les_b.nc", valid_time=SYNC_TIME,
                      seed=seed + 5, lon_center=lon_c, lat_center=lat_c)
    _write_simulation(root / "simulation" / "les_off.nc", valid_time=off_time,
                      seed=seed + 6, lon_center=lon_c, lat_center=lat_c)

    # --- sensor: high-rate samples straddling SYNC_TIME (some in-window, some
    #     out) at two in-region stations + one off-time file + off-location rows.
    in_window = SYNC_TIME + np.linspace(-600, 600, 20)      # within +/-30 min
    out_window = SYNC_TIME + np.array([5 * 3600.0, 6 * 3600.0])  # outside window
    times = np.concatenate([in_window, out_window])
    _write_sensor(root / "sensor" / "sonic_a.nc", times=times, seed=seed + 7,
                  lon=lon_c - 0.02, lat=lat_c + 0.01, alt=320.0)
    _write_sensor(root / "sensor" / "sonic_b.nc", times=in_window, seed=seed + 8,
                  lon=lon_c + 0.03, lat=lat_c - 0.02, alt=330.0)
    # off-time station: all samples far in time -> should be fully dropped
    _write_sensor(root / "sensor" / "sonic_off.nc", times=off_time + in_window - SYNC_TIME,
                  seed=seed + 9, lon=lon_c, lat=lat_c, alt=325.0)
    # off-location station: in-window times but well outside the HRRR bbox
    _write_sensor(root / "sensor" / "sonic_far.nc", times=in_window, seed=seed + 10,
                  lon=lon_c + 5.0, lat=lat_c + 5.0, alt=300.0)

    return {
        "sync_time": SYNC_TIME,
        "off_time": off_time,
        "region_lon": REGION_LON,
        "region_lat": REGION_LAT,
        "sensor_in_window_times": in_window.tolist(),
        "sensor_out_window_times": out_window.tolist(),
        "off_location_station": "sonic_far",
        "off_time_station": "sonic_off",
    }


if __name__ == "__main__":
    import sys
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("fixtures/generated")
    manifest = generate(dest)
    print(f"wrote phony data to {dest}")
    for k, v in manifest.items():
        print(f"  {k}: {v}")
