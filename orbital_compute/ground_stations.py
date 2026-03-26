from __future__ import annotations

"""Ground station network — contact windows for data uplink/downlink.

Satellites can only transfer data to/from Earth during ground station passes.
This constrains when results can be delivered and new jobs uploaded.
"""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Tuple

import numpy as np

from .orbit import Satellite, EARTH_RADIUS_KM, eci_to_lla


@dataclass
class GroundStation:
    """A ground station that can communicate with satellites."""
    name: str
    lat_deg: float
    lon_deg: float
    min_elevation_deg: float = 10.0    # Minimum elevation for contact
    uplink_mbps: float = 100.0         # Upload speed to satellite
    downlink_mbps: float = 500.0       # Download speed from satellite


@dataclass
class ContactWindow:
    """A period when a satellite can communicate with a ground station."""
    satellite_name: str
    station_name: str
    start: datetime
    end: datetime
    max_elevation_deg: float
    downlink_mb: float = 0.0  # Data that can be transferred

    @property
    def duration_seconds(self) -> float:
        return (self.end - self.start).total_seconds()


# Major ground station networks
DEFAULT_GROUND_STATIONS = [
    GroundStation("Svalbard", 78.23, 15.39, 5.0, 150, 1000),      # Norway — polar coverage
    GroundStation("Fairbanks", 64.86, -147.72, 10.0, 100, 500),    # Alaska
    GroundStation("Wallops", 37.94, -75.46, 10.0, 100, 500),       # Virginia
    GroundStation("Canberra", -35.40, 148.98, 10.0, 100, 500),     # Australia
    GroundStation("Bangalore", 13.03, 77.57, 10.0, 100, 500),      # India
    GroundStation("Hartebeest", -25.89, 27.69, 10.0, 100, 500),    # South Africa
    GroundStation("Santiago", -33.15, -70.67, 10.0, 100, 500),     # Chile
    GroundStation("Tromso", 69.66, 18.94, 5.0, 150, 1000),        # Norway
    GroundStation("McMurdo", -77.85, 166.67, 5.0, 50, 200),       # Antarctica
    GroundStation("Singapore", 1.35, 103.82, 10.0, 100, 500),     # Singapore
]


def _lla_to_ecef(lat_deg: float, lon_deg: float, alt_km: float = 0.0) -> np.ndarray:
    """Convert lat/lon/alt to ECEF (Earth-Centered Earth-Fixed)."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    r = EARTH_RADIUS_KM + alt_km
    x = r * math.cos(lat) * math.cos(lon)
    y = r * math.cos(lat) * math.sin(lon)
    z = r * math.sin(lat)
    return np.array([x, y, z])


def _ecef_to_eci(ecef: np.ndarray, dt: datetime) -> np.ndarray:
    """Convert ECEF to ECI by rotating by GMST."""
    jd = (dt - datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)).total_seconds() / 86400.0
    gmst_deg = (280.46061837 + 360.98564736629 * jd) % 360
    gmst = math.radians(gmst_deg)
    cos_g, sin_g = math.cos(gmst), math.sin(gmst)
    x = ecef[0] * cos_g - ecef[1] * sin_g
    y = ecef[0] * sin_g + ecef[1] * cos_g
    z = ecef[2]
    return np.array([x, y, z])


def elevation_angle(sat_pos_eci: np.ndarray, station: GroundStation, dt: datetime) -> float:
    """Calculate elevation angle of satellite as seen from ground station."""
    station_ecef = _lla_to_ecef(station.lat_deg, station.lon_deg)
    station_eci = _ecef_to_eci(station_ecef, dt)

    # Vector from station to satellite
    to_sat = sat_pos_eci - station_eci
    dist = np.linalg.norm(to_sat)

    if dist < 1.0:
        return 90.0

    # Local "up" direction at station (radial from Earth center)
    up = station_eci / np.linalg.norm(station_eci)

    # Elevation = 90° - angle between up and to_sat
    cos_angle = np.dot(to_sat, up) / dist
    angle_from_zenith = math.degrees(math.acos(max(-1, min(1, cos_angle))))
    elevation = 90.0 - angle_from_zenith

    return elevation


def find_contact_windows(satellite: Satellite, stations: List[GroundStation],
                         start: datetime, duration_hours: float,
                         step_seconds: float = 30.0) -> List[ContactWindow]:
    """Find all contact windows between a satellite and ground stations."""
    windows = []
    end = start + timedelta(hours=duration_hours)
    step = timedelta(seconds=step_seconds)

    for station in stations:
        current = start
        in_contact = False
        contact_start = None
        max_elev = 0.0

        while current <= end:
            pos = satellite.position_at(current)
            sat_eci = np.array([pos.x_km, pos.y_km, pos.z_km])
            elev = elevation_angle(sat_eci, station, current)

            if elev >= station.min_elevation_deg:
                if not in_contact:
                    contact_start = current
                    in_contact = True
                    max_elev = elev
                else:
                    max_elev = max(max_elev, elev)
            elif in_contact:
                duration_s = (current - contact_start).total_seconds()
                downlink_mb = station.downlink_mbps * duration_s / 8  # bits to bytes
                windows.append(ContactWindow(
                    satellite_name=satellite.name,
                    station_name=station.name,
                    start=contact_start,
                    end=current,
                    max_elevation_deg=round(max_elev, 1),
                    downlink_mb=round(downlink_mb, 1),
                ))
                in_contact = False

            current += step

        # Close any open window
        if in_contact and contact_start:
            duration_s = (current - contact_start).total_seconds()
            downlink_mb = station.downlink_mbps * duration_s / 8
            windows.append(ContactWindow(
                satellite_name=satellite.name,
                station_name=station.name,
                start=contact_start,
                end=current,
                max_elevation_deg=round(max_elev, 1),
                downlink_mb=round(downlink_mb, 1),
            ))

    windows.sort(key=lambda w: w.start)
    return windows
