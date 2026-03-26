from __future__ import annotations

"""Orbital mechanics — satellite position, velocity, and eclipse prediction.

Uses SGP4 for TLE propagation. Eclipse detection uses Earth's cylindrical shadow model.
"""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
from sgp4.api import Satrec, jday
from sgp4.conveniences import dump_satrec


# Constants
EARTH_RADIUS_KM = 6371.0
SUN_RADIUS_KM = 696340.0
AU_KM = 149597870.7


@dataclass
class SatPosition:
    """Satellite position in ECI (Earth-Centered Inertial) frame."""
    x_km: float
    y_km: float
    z_km: float
    vx_kms: float
    vy_kms: float
    vz_kms: float
    altitude_km: float
    in_eclipse: bool
    lat_deg: float
    lon_deg: float


def sun_position_eci(dt: datetime) -> np.ndarray:
    """Approximate sun position in ECI frame.

    Uses a simplified solar position model accurate to ~1 degree.
    Good enough for eclipse prediction — we don't need arcsecond precision.
    """
    # Days since J2000.0
    jd = (dt - datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)).total_seconds() / 86400.0

    # Mean longitude and anomaly (degrees)
    L0 = (280.460 + 0.9856474 * jd) % 360
    M = math.radians((357.528 + 0.9856003 * jd) % 360)

    # Ecliptic longitude
    ecliptic_lon = math.radians(L0 + 1.915 * math.sin(M) + 0.020 * math.sin(2 * M))

    # Obliquity of ecliptic
    obliquity = math.radians(23.439 - 0.0000004 * jd)

    # Distance (AU)
    R_au = 1.00014 - 0.01671 * math.cos(M) - 0.00014 * math.cos(2 * M)
    R_km = R_au * AU_KM

    # ECI coordinates
    x = R_km * math.cos(ecliptic_lon)
    y = R_km * math.cos(obliquity) * math.sin(ecliptic_lon)
    z = R_km * math.sin(obliquity) * math.sin(ecliptic_lon)

    return np.array([x, y, z])


def is_in_eclipse(sat_pos_km: np.ndarray, sun_pos_km: np.ndarray) -> bool:
    """Check if satellite is in Earth's shadow using cylindrical shadow model.

    Simple and fast — treats Earth's shadow as a cylinder (ignores penumbra).
    Accurate enough for power scheduling.
    """
    # Vector from Earth to satellite
    sat = sat_pos_km

    # Vector from Earth to Sun
    sun_dir = sun_pos_km / np.linalg.norm(sun_pos_km)

    # Project satellite position onto sun direction
    proj = np.dot(sat, sun_dir)

    # If satellite is on the sun-side of Earth, it's not in eclipse
    if proj > 0:
        return False

    # Perpendicular distance from sat to Earth-Sun line
    perp = sat - proj * sun_dir
    perp_dist = np.linalg.norm(perp)

    return perp_dist < EARTH_RADIUS_KM


def eci_to_lla(pos_km: np.ndarray, dt: datetime) -> tuple[float, float, float]:
    """Convert ECI position to latitude, longitude, altitude.

    Uses a simplified conversion (ignores Earth oblateness).
    """
    x, y, z = pos_km

    # Altitude
    r = math.sqrt(x**2 + y**2 + z**2)
    alt = r - EARTH_RADIUS_KM

    # Latitude (geodetic approximation)
    lat = math.degrees(math.asin(z / r))

    # Greenwich Mean Sidereal Time
    jd_val = (dt - datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)).total_seconds() / 86400.0
    gmst = (280.46061837 + 360.98564736629 * jd_val) % 360

    # Longitude
    lon = math.degrees(math.atan2(y, x)) - gmst
    lon = ((lon + 180) % 360) - 180  # Normalize to [-180, 180]

    return lat, lon, alt


class Satellite:
    """A compute satellite with orbital state."""

    def __init__(self, name: str, tle_line1: str, tle_line2: str):
        self.name = name
        self.tle_line1 = tle_line1
        self.tle_line2 = tle_line2
        self.satrec = Satrec.twoline2rv(tle_line1, tle_line2)

    def position_at(self, dt: datetime) -> SatPosition:
        """Get satellite position at a given time."""
        jd, fr = jday(dt.year, dt.month, dt.day,
                       dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)

        error, pos, vel = self.satrec.sgp4(jd, fr)
        if error != 0:
            raise ValueError(f"SGP4 error {error} for {self.name}")

        pos_km = np.array(pos)
        vel_kms = np.array(vel)

        sun_pos = sun_position_eci(dt)
        eclipse = is_in_eclipse(pos_km, sun_pos)

        lat, lon, alt = eci_to_lla(pos_km, dt)

        return SatPosition(
            x_km=pos[0], y_km=pos[1], z_km=pos[2],
            vx_kms=vel[0], vy_kms=vel[1], vz_kms=vel[2],
            altitude_km=alt,
            in_eclipse=eclipse,
            lat_deg=lat, lon_deg=lon,
        )


def predict_eclipse_windows(sat: Satellite, start: datetime, duration_hours: float,
                             step_seconds: float = 60.0) -> list[tuple[datetime, datetime]]:
    """Predict eclipse windows for a satellite over a time range.

    Returns list of (eclipse_start, eclipse_end) tuples.
    """
    windows = []
    current = start
    end = start + timedelta(hours=duration_hours)
    step = timedelta(seconds=step_seconds)

    in_eclipse = False
    eclipse_start = None

    while current <= end:
        pos = sat.position_at(current)

        if pos.in_eclipse and not in_eclipse:
            eclipse_start = current
            in_eclipse = True
        elif not pos.in_eclipse and in_eclipse:
            windows.append((eclipse_start, current))
            in_eclipse = False

        current += step

    if in_eclipse and eclipse_start:
        windows.append((eclipse_start, current))

    return windows


# Pre-built constellation helpers

def starlink_shell_1_sample(n_sats: int = 12) -> list[Satellite]:
    """Generate sample satellites in a Starlink-like shell (550 km, 53° inclination).

    Uses synthetic TLEs — not real Starlink satellites.
    Good for simulation/testing.
    """
    satellites = []
    # 550 km altitude, 53 deg inclination, circular orbit
    # Mean motion ~15.05 rev/day for 550km
    for i in range(n_sats):
        raan = (360.0 / n_sats) * i  # Spread RAAN evenly
        mean_anomaly = (360.0 / n_sats) * i * 7 % 360  # Offset within orbit

        # Build TLE
        # Line 1: catalog number, epoch, etc (simplified)
        line1 = f"1 {50000+i:05d}U 24001A   26085.00000000  .00000000  00000-0  00000-0 0  999{'0' if (i+1)%10==0 else str((i+1)%10)}"
        # Line 2: inclination, RAAN, eccentricity, arg perigee, mean anomaly, mean motion
        line2 = f"2 {50000+i:05d}  53.0000 {raan:8.4f} 0001000   0.0000 {mean_anomaly:8.4f} 15.05000000    0{'0' if (i+1)%10==0 else str((i+1)%10)}"

        # Fix TLE checksums
        line1 = _fix_tle_checksum(line1)
        line2 = _fix_tle_checksum(line2)

        satellites.append(Satellite(f"SAT-{i:03d}", line1, line2))

    return satellites


def _fix_tle_checksum(line: str) -> str:
    """Calculate and append TLE checksum digit."""
    checksum = 0
    for ch in line[:68]:
        if ch.isdigit():
            checksum += int(ch)
        elif ch == '-':
            checksum += 1
    line = line[:68] + str(checksum % 10)
    return line


if __name__ == "__main__":
    from datetime import datetime, timezone
    print("=" * 60)
    print("  ORBITAL MECHANICS DEMO")
    print("=" * 60)
    sats = starlink_shell_1_sample(6)
    t = datetime(2026, 3, 26, 12, 0, 0, tzinfo=timezone.utc)
    print(f"\n  6 satellites at 550km, 53° inclination")
    print(f"  Time: {t.isoformat()}\n")
    print(f"  {'Satellite':<10} {'Lat':>7} {'Lon':>8} {'Alt km':>7} {'Eclipse':>8}")
    print(f"  {'-'*40}")
    for sat in sats:
        pos = sat.position_at(t)
        print(f"  {sat.name:<10} {pos.lat_deg:>6.1f}° {pos.lon_deg:>7.1f}° "
              f"{pos.altitude_km:>6.0f} {'YES' if pos.in_eclipse else 'no':>8}")
    windows = predict_eclipse_windows(sats[0], t, 6.0)
    print(f"\n  Eclipse windows for {sats[0].name} (next 6h):")
    for i, (start, end) in enumerate(windows[:4]):
        dur = (end - start).total_seconds() / 60
        print(f"    {start.strftime('%H:%M')} → {end.strftime('%H:%M')} ({dur:.0f} min)")
    print(f"\n{'=' * 60}")
