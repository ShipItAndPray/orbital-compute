"""Industry-standard data format support for the orbital compute simulator.

Supports:
- TLE (Two-Line Element) import/export with checksum validation
- CCSDS OEM (Orbit Ephemeris Message) export
- AGI STK ephemeris (.e) export
- JSON Schema for simulation results
- Ground station database import/export (CCSDS-style)
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Union

from .orbit import Satellite, SatPosition, EARTH_RADIUS_KM, _fix_tle_checksum
from .ground_stations import GroundStation, DEFAULT_GROUND_STATIONS


# ============================================================================
# TLE Import / Export
# ============================================================================

def tle_checksum(line: str) -> int:
    """Calculate TLE checksum for a line (mod-10 of digit sum, '-' counts as 1)."""
    total = 0
    for ch in line[:68]:
        if ch.isdigit():
            total += int(ch)
        elif ch == '-':
            total += 1
    return total % 10


def validate_tle_checksum(line: str) -> bool:
    """Validate TLE line checksum (last digit)."""
    if len(line) < 69:
        return False
    expected = int(line[68])
    return tle_checksum(line) == expected


def validate_tle(line1: str, line2: str) -> List[str]:
    """Validate a TLE pair. Returns list of errors (empty = valid)."""
    errors = []
    if len(line1) != 69:
        errors.append(f"Line 1 length {len(line1)}, expected 69")
    if len(line2) != 69:
        errors.append(f"Line 2 length {len(line2)}, expected 69")
    if not line1.startswith("1 "):
        errors.append("Line 1 must start with '1 '")
    if not line2.startswith("2 "):
        errors.append("Line 2 must start with '2 '")
    if len(line1) >= 69 and not validate_tle_checksum(line1):
        errors.append(f"Line 1 checksum invalid (got {line1[68]}, expected {tle_checksum(line1)})")
    if len(line2) >= 69 and not validate_tle_checksum(line2):
        errors.append(f"Line 2 checksum invalid (got {line2[68]}, expected {tle_checksum(line2)})")
    if len(errors) == 0:
        # Cross-check catalog numbers
        cat1 = line1[2:7].strip()
        cat2 = line2[2:7].strip()
        if cat1 != cat2:
            errors.append(f"Catalog number mismatch: {cat1} vs {cat2}")
    return errors


def parse_tle(text: str) -> List[Tuple[str, str, str]]:
    """Parse TLE text into list of (name, line1, line2) tuples.

    Handles both 2-line format (no name) and 3-line format (with name).
    """
    lines = [l.rstrip() for l in text.strip().split('\n') if l.strip()]
    results = []
    i = 0

    while i < len(lines):
        # Check if current line is line 1 of a TLE
        if lines[i].startswith('1 ') and i + 1 < len(lines) and lines[i + 1].startswith('2 '):
            # 2-line format (no name)
            line1 = lines[i]
            line2 = lines[i + 1]
            cat_num = line1[2:7].strip()
            results.append((f"SAT-{cat_num}", line1, line2))
            i += 2
        elif i + 2 < len(lines) and lines[i + 1].startswith('1 ') and lines[i + 2].startswith('2 '):
            # 3-line format (with name)
            name = lines[i].strip()
            line1 = lines[i + 1]
            line2 = lines[i + 2]
            results.append((name, line1, line2))
            i += 3
        else:
            i += 1  # Skip unrecognized line

    return results


def parse_tle_to_satellites(text: str) -> List[Satellite]:
    """Parse TLE text and return Satellite objects."""
    parsed = parse_tle(text)
    satellites = []
    for name, line1, line2 in parsed:
        try:
            satellites.append(Satellite(name, line1, line2))
        except Exception:
            pass  # Skip malformed TLEs
    return satellites


def export_tle(satellites: List[Satellite], include_names: bool = True) -> str:
    """Export satellites as TLE text (3-line format with names by default)."""
    lines = []
    for sat in satellites:
        if include_names:
            lines.append(sat.name)
        lines.append(sat.tle_line1)
        lines.append(sat.tle_line2)
    return '\n'.join(lines) + '\n'


def generate_synthetic_tle(
    name: str,
    catalog_number: int,
    inclination_deg: float,
    raan_deg: float,
    eccentricity: float,
    arg_perigee_deg: float,
    mean_anomaly_deg: float,
    mean_motion_rev_day: float,
    epoch_year: int = 26,
    epoch_day: float = 85.0,
) -> Tuple[str, str]:
    """Generate a synthetic TLE from orbital elements.

    Returns (line1, line2) with valid checksums.
    """
    line1 = "1 {:05d}U 26001A   {:02d}{:012.8f}  .00000000  00000-0  00000-0 0  9990".format(
        catalog_number, epoch_year, epoch_day
    )
    # Eccentricity in TLE format: 7 digits, no leading "0."
    ecc_str = "{:.7f}".format(eccentricity)[2:]  # strip "0."
    line2 = "2 {:05d} {:8.4f} {:8.4f} {} {:8.4f} {:8.4f} {:11.8f}    00".format(
        catalog_number, inclination_deg, raan_deg, ecc_str,
        arg_perigee_deg, mean_anomaly_deg, mean_motion_rev_day
    )
    line1 = _fix_tle_checksum(line1)
    line2 = _fix_tle_checksum(line2)
    return line1, line2


# ============================================================================
# CCSDS OEM (Orbit Ephemeris Message) Export
# ============================================================================

def export_oem(
    satellite: Satellite,
    start: datetime,
    duration_hours: float,
    step_seconds: float = 60.0,
    originator: str = "ORBITAL-COMPUTE-SIM",
    object_id: Optional[str] = None,
    center_name: str = "EARTH",
    ref_frame: str = "EME2000",
    time_system: str = "UTC",
) -> str:
    """Export satellite ephemeris in CCSDS OEM 2.0 text format.

    Reference: CCSDS 502.0-B-3 (Orbit Data Messages)

    The OEM format is the industry standard for exchanging orbit ephemeris data
    between space agencies and operators.
    """
    creation_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    start_str = start.strftime("%Y-%m-%dT%H:%M:%S.%f")[:23]
    end_time = start + timedelta(hours=duration_hours)
    end_str = end_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:23]
    obj_id = object_id or satellite.name

    header = (
        "CCSDS_OEM_VERS = 2.0\n"
        f"CREATION_DATE = {creation_date}\n"
        f"ORIGINATOR = {originator}\n"
        "\n"
        "META_START\n"
        f"OBJECT_NAME = {satellite.name}\n"
        f"OBJECT_ID = {obj_id}\n"
        f"CENTER_NAME = {center_name}\n"
        f"REF_FRAME = {ref_frame}\n"
        f"TIME_SYSTEM = {time_system}\n"
        f"START_TIME = {start_str}\n"
        f"STOP_TIME = {end_str}\n"
        "META_STOP\n"
        "\n"
        "COMMENT  Position (km) and Velocity (km/s) in EME2000 frame\n"
        "COMMENT  Generated by orbital-compute simulator\n"
    )

    data_lines = []
    current = start
    step = timedelta(seconds=step_seconds)
    while current <= end_time:
        pos = satellite.position_at(current)
        time_str = current.strftime("%Y-%m-%dT%H:%M:%S.%f")[:23]
        data_lines.append(
            f"{time_str}  {pos.x_km:14.6f}  {pos.y_km:14.6f}  {pos.z_km:14.6f}"
            f"  {pos.vx_kms:12.9f}  {pos.vy_kms:12.9f}  {pos.vz_kms:12.9f}"
        )
        current += step

    return header + '\n'.join(data_lines) + '\n'


# ============================================================================
# AGI STK Ephemeris (.e) Export
# ============================================================================

def export_stk_ephemeris(
    satellite: Satellite,
    start: datetime,
    duration_hours: float,
    step_seconds: float = 60.0,
    coord_system: str = "J2000",
) -> str:
    """Export satellite ephemeris in AGI STK .e format.

    This lets users import simulation results into STK (Systems Tool Kit),
    the industry-standard tool for space mission analysis.
    """
    end_time = start + timedelta(hours=duration_hours)

    # Collect ephemeris points
    points = []
    current = start
    step = timedelta(seconds=step_seconds)

    # Reference epoch for STK seconds
    epoch = start

    while current <= end_time:
        pos = satellite.position_at(current)
        elapsed_s = (current - epoch).total_seconds()
        # STK uses km and km/s
        points.append((elapsed_s, pos.x_km, pos.y_km, pos.z_km,
                        pos.vx_kms, pos.vy_kms, pos.vz_kms))
        current += step

    epoch_str = epoch.strftime("%d %b %Y %H:%M:%S.%f")[:24]

    lines = [
        "stk.v.12.0",
        "",
        "BEGIN Ephemeris",
        "",
        f"    NumberOfEphemerisPoints  {len(points)}",
        "",
        f"    ScenarioEpoch           {epoch_str}",
        f"    InterpolationMethod     Lagrange",
        f"    InterpolationOrder      7",
        f"    CentralBody             Earth",
        f"    CoordinateSystem        {coord_system}",
        "",
        "    EphemerisTimePosVel",
        "",
    ]

    for t, x, y, z, vx, vy, vz in points:
        # STK format: time(s) x(km) y(km) z(km) vx(km/s) vy(km/s) vz(km/s)
        lines.append(
            f"    {t:14.6f}  {x:14.6f}  {y:14.6f}  {z:14.6f}"
            f"  {vx:14.9f}  {vy:14.9f}  {vz:14.9f}"
        )

    lines.extend([
        "",
        "END Ephemeris",
        "",
    ])

    return '\n'.join(lines)


# ============================================================================
# JSON Schema for Simulation Results
# ============================================================================

SIMULATION_RESULT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://github.com/orbital-compute/sim-result-schema/v1",
    "title": "Orbital Compute Simulation Result",
    "description": "Output schema for the orbital compute constellation simulator",
    "type": "object",
    "required": ["meta", "config", "scheduler", "fleet_utilization_pct",
                  "total_compute_hours", "satellite_details"],
    "properties": {
        "meta": {
            "type": "object",
            "description": "Simulation metadata",
            "required": ["version", "created_at", "simulator"],
            "properties": {
                "version": {"type": "string", "description": "Schema version"},
                "created_at": {"type": "string", "format": "date-time"},
                "simulator": {"type": "string", "const": "orbital-compute"},
                "simulator_version": {"type": "string"},
            },
        },
        "config": {
            "type": "object",
            "description": "Simulation configuration",
            "required": ["n_satellites", "sim_hours", "n_jobs"],
            "properties": {
                "n_satellites": {"type": "integer", "minimum": 1},
                "sim_hours": {"type": "number", "minimum": 0},
                "n_jobs": {"type": "integer", "minimum": 0},
                "time_step_seconds": {"type": "number", "minimum": 1},
                "start_time": {"type": "string", "format": "date-time"},
                "constellation": {"type": "string"},
            },
        },
        "scheduler": {
            "type": "object",
            "required": ["total_jobs", "completed", "running", "queued"],
            "properties": {
                "total_jobs": {"type": "integer", "minimum": 0},
                "completed": {"type": "integer", "minimum": 0},
                "running": {"type": "integer", "minimum": 0},
                "queued": {"type": "integer", "minimum": 0},
                "preempted": {"type": "integer", "minimum": 0},
                "idle_steps": {"type": "integer", "minimum": 0},
                "charge_steps": {"type": "integer", "minimum": 0},
            },
        },
        "fleet_utilization_pct": {
            "type": "number", "minimum": 0, "maximum": 100,
            "description": "Percentage of total constellation time spent computing",
        },
        "total_compute_hours": {
            "type": "number", "minimum": 0,
            "description": "Total compute hours delivered across all satellites",
        },
        "satellite_details": {
            "type": "object",
            "description": "Per-satellite performance metrics",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "compute_pct": {"type": "number", "minimum": 0, "maximum": 100},
                    "eclipse_pct": {"type": "number", "minimum": 0, "maximum": 100},
                    "avg_battery_pct": {"type": "number"},
                    "avg_temp_c": {"type": "number"},
                    "max_temp_c": {"type": "number"},
                    "min_battery_pct": {"type": "number"},
                    "contact_windows": {"type": "integer", "minimum": 0},
                    "total_contact_minutes": {"type": "number", "minimum": 0},
                },
            },
        },
        "completed_jobs": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["job_id"],
                "properties": {
                    "job_id": {"type": "string"},
                    "satellite": {"type": "string"},
                    "duration_s": {"type": "number"},
                    "power_w": {"type": "number"},
                },
            },
        },
        "preemption_events": {"type": "integer", "minimum": 0},
    },
}


def get_simulation_schema() -> dict:
    """Return the JSON Schema for simulation results."""
    return SIMULATION_RESULT_SCHEMA


def wrap_results_with_metadata(results: dict, config: Optional[dict] = None) -> dict:
    """Wrap simulation results with metadata for schema compliance."""
    meta = {
        "version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "simulator": "orbital-compute",
        "simulator_version": "0.1.0",
    }
    output = {"meta": meta}
    output.update(results)
    return output


def validate_results_against_schema(results: dict) -> List[str]:
    """Basic validation of simulation results against schema (no jsonschema dep).

    Returns list of errors (empty = valid).
    """
    errors = []
    schema = SIMULATION_RESULT_SCHEMA
    required_top = schema.get("required", [])

    for field in required_top:
        if field not in results:
            errors.append(f"Missing required field: {field}")

    if "fleet_utilization_pct" in results:
        val = results["fleet_utilization_pct"]
        if not isinstance(val, (int, float)):
            errors.append("fleet_utilization_pct must be a number")
        elif val < 0 or val > 100:
            errors.append(f"fleet_utilization_pct={val} out of range [0, 100]")

    if "satellite_details" in results:
        if not isinstance(results["satellite_details"], dict):
            errors.append("satellite_details must be an object")

    if "completed_jobs" in results:
        if not isinstance(results["completed_jobs"], list):
            errors.append("completed_jobs must be an array")
        else:
            for i, job in enumerate(results["completed_jobs"]):
                if "job_id" not in job:
                    errors.append(f"completed_jobs[{i}] missing job_id")

    return errors


# ============================================================================
# Ground Station Database (CCSDS-style)
# ============================================================================

# Standard ground station networks
NASA_NEN_STATIONS = [
    GroundStation("Svalbard-NEN", 78.23, 15.39, 5.0, 150, 1000),
    GroundStation("Wallops-NEN", 37.94, -75.46, 10.0, 100, 500),
    GroundStation("Fairbanks-NEN", 64.86, -147.72, 10.0, 100, 500),
    GroundStation("McMurdo-NEN", -77.85, 166.67, 5.0, 50, 200),
    GroundStation("Singapore-NEN", 1.35, 103.82, 10.0, 100, 500),
    GroundStation("Hartebeest-NEN", -25.89, 27.69, 10.0, 100, 500),
]

ESA_ESTRACK_STATIONS = [
    GroundStation("Kiruna-EST", 67.86, 20.96, 5.0, 150, 1000),
    GroundStation("Redu-EST", 50.00, 5.15, 10.0, 100, 500),
    GroundStation("Cebreros-EST", 40.45, -4.37, 10.0, 100, 500),
    GroundStation("Malargue-EST", -35.78, -69.40, 10.0, 100, 500),
    GroundStation("NewNorcia-EST", -31.05, 116.19, 10.0, 100, 500),
    GroundStation("Kourou-EST", 5.25, -52.93, 10.0, 100, 500),
]

KSAT_STATIONS = [
    GroundStation("Svalbard-KSAT", 78.23, 15.39, 5.0, 200, 1200),
    GroundStation("Tromso-KSAT", 69.66, 18.94, 5.0, 150, 1000),
    GroundStation("Grimstad-KSAT", 58.34, 8.36, 10.0, 100, 500),
    GroundStation("Punta-Arenas-KSAT", -53.15, -70.92, 5.0, 150, 1000),
    GroundStation("Inuvik-KSAT", 68.36, -133.72, 10.0, 100, 500),
    GroundStation("Singapore-KSAT", 1.35, 103.82, 10.0, 100, 500),
    GroundStation("Dubai-KSAT", 25.27, 55.30, 10.0, 100, 500),
    GroundStation("Hawaii-KSAT", 19.82, -155.47, 10.0, 100, 500),
]

STATION_NETWORKS = {
    "nasa_nen": NASA_NEN_STATIONS,
    "esa_estrack": ESA_ESTRACK_STATIONS,
    "ksat": KSAT_STATIONS,
    "default": DEFAULT_GROUND_STATIONS,
}


def export_ground_stations_ccsds(stations: List[GroundStation],
                                  network_name: str = "CUSTOM") -> str:
    """Export ground stations in a CCSDS-style text format.

    Based on CCSDS 503.0-B-2 (Tracking Data Message) station catalog conventions.
    """
    creation = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    lines = [
        "CCSDS_TDM_VERS = 2.0",
        f"CREATION_DATE = {creation}",
        f"ORIGINATOR = ORBITAL-COMPUTE-SIM",
        "",
        f"COMMENT Ground station network: {network_name}",
        f"COMMENT Number of stations: {len(stations)}",
        "",
    ]

    for station in stations:
        lines.extend([
            "META_START",
            f"    PARTICIPANT_1 = {station.name}",
            f"    PARTICIPANT_TYPE = GROUND_STATION",
            f"    LATITUDE_DEG = {station.lat_deg:.4f}",
            f"    LONGITUDE_DEG = {station.lon_deg:.4f}",
            f"    MIN_ELEVATION_DEG = {station.min_elevation_deg:.1f}",
            f"    UPLINK_RATE_MBPS = {station.uplink_mbps:.1f}",
            f"    DOWNLINK_RATE_MBPS = {station.downlink_mbps:.1f}",
            "META_STOP",
            "",
        ])

    return '\n'.join(lines)


def import_ground_stations_ccsds(text: str) -> List[GroundStation]:
    """Import ground stations from CCSDS-style text format."""
    stations = []
    current = {}

    for line in text.split('\n'):
        line = line.strip()
        if line == "META_START":
            current = {}
        elif line == "META_STOP":
            if "name" in current:
                stations.append(GroundStation(
                    name=current.get("name", "UNKNOWN"),
                    lat_deg=current.get("lat", 0.0),
                    lon_deg=current.get("lon", 0.0),
                    min_elevation_deg=current.get("min_elev", 10.0),
                    uplink_mbps=current.get("uplink", 100.0),
                    downlink_mbps=current.get("downlink", 500.0),
                ))
            current = {}
        elif '=' in line:
            key, val = line.split('=', 1)
            key = key.strip()
            val = val.strip()
            if key == "PARTICIPANT_1":
                current["name"] = val
            elif key == "LATITUDE_DEG":
                current["lat"] = float(val)
            elif key == "LONGITUDE_DEG":
                current["lon"] = float(val)
            elif key == "MIN_ELEVATION_DEG":
                current["min_elev"] = float(val)
            elif key == "UPLINK_RATE_MBPS":
                current["uplink"] = float(val)
            elif key == "DOWNLINK_RATE_MBPS":
                current["downlink"] = float(val)

    return stations


def get_station_network(network: str) -> List[GroundStation]:
    """Get a predefined ground station network.

    Available: 'nasa_nen', 'esa_estrack', 'ksat', 'default'
    """
    if network not in STATION_NETWORKS:
        raise ValueError(f"Unknown network: {network}. Available: {list(STATION_NETWORKS.keys())}")
    return STATION_NETWORKS[network]


# ============================================================================
# Self-test
# ============================================================================

def _self_test():
    """Test all format import/export functions."""
    from .orbit import starlink_shell_1_sample

    print("=" * 60)
    print("  ORBITAL COMPUTE -- FORMAT SUPPORT SELF-TEST")
    print("=" * 60)

    errors = []

    # --- TLE Tests ---
    print("\n  [TLE] Testing TLE import/export...")

    sats = starlink_shell_1_sample(4)

    # Export
    tle_text = export_tle(sats, include_names=True)
    assert "SAT-" in tle_text, "TLE export missing satellite names"
    tle_lines = [l for l in tle_text.strip().split('\n') if l.strip()]
    assert len(tle_lines) == 12, f"Expected 12 TLE lines (3 per sat), got {len(tle_lines)}"

    # Re-import
    parsed = parse_tle(tle_text)
    assert len(parsed) == 4, f"Expected 4 parsed TLEs, got {len(parsed)}"

    reimported = parse_tle_to_satellites(tle_text)
    assert len(reimported) == 4, f"Expected 4 reimported sats, got {len(reimported)}"

    # Checksum validation
    for sat in sats:
        assert validate_tle_checksum(sat.tle_line1), f"Line 1 checksum failed for {sat.name}"
        assert validate_tle_checksum(sat.tle_line2), f"Line 2 checksum failed for {sat.name}"

    # Full validation
    for sat in sats:
        errs = validate_tle(sat.tle_line1, sat.tle_line2)
        if errs:
            errors.append(f"TLE validation failed for {sat.name}: {errs}")

    # 2-line format (no names)
    tle_no_names = export_tle(sats, include_names=False)
    parsed2 = parse_tle(tle_no_names)
    assert len(parsed2) == 4, f"2-line parse: expected 4, got {len(parsed2)}"

    # Synthetic TLE generation
    line1, line2 = generate_synthetic_tle(
        "TEST-SAT", 99999, 53.0, 120.0, 0.0001, 0.0, 180.0, 15.05
    )
    errs = validate_tle(line1, line2)
    if errs:
        errors.append(f"Synthetic TLE validation failed: {errs}")

    print("    TLE export/import:    OK")
    print("    TLE checksums:        OK")
    print("    TLE validation:       OK")
    print("    Synthetic TLE:        OK")

    # --- CCSDS OEM Test ---
    print("\n  [OEM] Testing CCSDS OEM export...")

    sat = sats[0]
    start = datetime(2026, 3, 26, 0, 0, 0, tzinfo=timezone.utc)
    oem = export_oem(sat, start, duration_hours=0.5, step_seconds=60.0)

    assert "CCSDS_OEM_VERS = 2.0" in oem, "OEM missing version header"
    assert "META_START" in oem, "OEM missing META_START"
    assert "META_STOP" in oem, "OEM missing META_STOP"
    assert "EME2000" in oem, "OEM missing reference frame"
    assert sat.name in oem, "OEM missing satellite name"

    # Count data lines (non-header lines with position data)
    data_lines = [l for l in oem.split('\n') if l.strip() and
                  l.strip()[0:4].startswith('202')]
    expected_points = int(0.5 * 3600 / 60) + 1
    assert len(data_lines) == expected_points, \
        f"OEM data points: expected {expected_points}, got {len(data_lines)}"

    print(f"    OEM header:           OK")
    print(f"    OEM data points:      OK ({len(data_lines)} points)")
    print(f"    OEM sample line:      {data_lines[0][:60]}...")

    # --- STK Ephemeris Test ---
    print("\n  [STK] Testing STK ephemeris export...")

    stk = export_stk_ephemeris(sat, start, duration_hours=0.5, step_seconds=60.0)

    assert "stk.v.12.0" in stk, "STK missing version"
    assert "BEGIN Ephemeris" in stk, "STK missing BEGIN"
    assert "END Ephemeris" in stk, "STK missing END"
    assert "EphemerisTimePosVel" in stk, "STK missing data type"
    assert f"NumberOfEphemerisPoints  {expected_points}" in stk, "STK wrong point count"

    print(f"    STK header:           OK")
    print(f"    STK structure:        OK ({expected_points} points)")

    # --- JSON Schema Test ---
    print("\n  [JSON] Testing JSON schema...")

    schema = get_simulation_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert "satellite_details" in schema["properties"]

    # Test validation
    good_result = {
        "meta": {"version": "1.0.0", "created_at": "2026-03-26T00:00:00",
                 "simulator": "orbital-compute"},
        "config": {"n_satellites": 6, "sim_hours": 6.0, "n_jobs": 20},
        "scheduler": {"total_jobs": 20, "completed": 15, "running": 2, "queued": 3},
        "fleet_utilization_pct": 45.2,
        "total_compute_hours": 8.5,
        "satellite_details": {},
        "completed_jobs": [{"job_id": "JOB-0001"}],
        "preemption_events": 3,
    }
    errs = validate_results_against_schema(good_result)
    assert len(errs) == 0, f"Valid result flagged errors: {errs}"

    # Test with bad data
    bad_result = {"fleet_utilization_pct": 150.0}  # missing fields + bad value
    errs = validate_results_against_schema(bad_result)
    assert len(errs) > 0, "Bad result should have errors"

    # Test metadata wrapping
    wrapped = wrap_results_with_metadata(good_result)
    assert wrapped["meta"]["simulator"] == "orbital-compute"

    print(f"    Schema definition:    OK")
    print(f"    Result validation:    OK")
    print(f"    Metadata wrapping:    OK")

    # --- Ground Station Database Test ---
    print("\n  [GS]  Testing ground station database...")

    for net_name in ["nasa_nen", "esa_estrack", "ksat"]:
        stations = get_station_network(net_name)
        assert len(stations) > 0, f"Empty network: {net_name}"
        print(f"    {net_name:15s}  {len(stations)} stations")

    # Export/import roundtrip
    nen = get_station_network("nasa_nen")
    exported = export_ground_stations_ccsds(nen, "NASA_NEN")
    assert "CCSDS_TDM_VERS" in exported
    assert "NASA_NEN" in exported

    reimported = import_ground_stations_ccsds(exported)
    assert len(reimported) == len(nen), \
        f"Roundtrip: exported {len(nen)}, reimported {len(reimported)}"

    for orig, re in zip(nen, reimported):
        assert orig.name == re.name, f"Name mismatch: {orig.name} vs {re.name}"
        assert abs(orig.lat_deg - re.lat_deg) < 0.01, f"Lat mismatch for {orig.name}"
        assert abs(orig.lon_deg - re.lon_deg) < 0.01, f"Lon mismatch for {orig.name}"

    print(f"    CCSDS export:         OK")
    print(f"    CCSDS roundtrip:      OK ({len(reimported)} stations)")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    if errors:
        print(f"  FAIL -- {len(errors)} error(s)")
        for e in errors:
            print(f"    {e}")
    else:
        print("  PASS -- all format tests passed")

        # Print sample outputs
        print(f"\n  Sample OEM output (first 8 lines):")
        for line in oem.split('\n')[:8]:
            print(f"    {line}")

        print(f"\n  Sample STK output (first 8 lines):")
        for line in stk.split('\n')[:8]:
            print(f"    {line}")

        print(f"\n  Sample ground station CCSDS (first 10 lines):")
        for line in exported.split('\n')[:10]:
            print(f"    {line}")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    _self_test()
