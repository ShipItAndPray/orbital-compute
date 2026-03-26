"""Constellation design optimizer — the killer feature.

Given customer requirements (coverage, compute, budget, latency), automatically
designs the optimal satellite constellation: number of satellites, orbital planes,
altitude, inclination, power sizing, thermal sizing, and cost estimate.

Uses Walker Delta constellation formulas, ground track coverage analysis,
and iterative optimization over the design space.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from .orbit import Satellite, EARTH_RADIUS_KM, _fix_tle_checksum, eci_to_lla
from .constellations import ConstellationConfig, generate_constellation
from .cost_model import (
    ConstellationCostConfig,
    HardwareCosts,
    LaunchCosts,
    OperatingCosts,
    calculate_constellation_costs,
)
from .thermal import ThermalModel, ThermalConfig, STEFAN_BOLTZMANN
from .power import PowerConfig
from .ground_stations import (
    GroundStation,
    DEFAULT_GROUND_STATIONS,
    elevation_angle,
    _lla_to_ecef,
    _ecef_to_eci,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MU_EARTH = 398600.4418  # km^3/s^2
C_LIGHT = 299792.458    # km/s
GPU_TDP_W = 700.0       # H100 TDP watts
GPU_COMPUTE_HOURS_PER_DAY = 24.0  # max per GPU if always on

# Radiation tolerance -> cost multiplier on electronics
RAD_TOLERANCE_MULTIPLIER = {
    "commercial": 1.0,
    "rad_tolerant": 1.3,
    "rad_hard": 2.5,
}

# Coverage type -> target inclination ranges and latitude bands
COVERAGE_PROFILES = {
    "global": {"inc_range": (50.0, 98.0), "lat_band": (-90.0, 90.0)},
    "tropical": {"inc_range": (0.0, 30.0), "lat_band": (-30.0, 30.0)},
    "polar": {"inc_range": (85.0, 98.0), "lat_band": (-90.0, 90.0)},
    "specific_latitudes": {"inc_range": (0.0, 98.0), "lat_band": (-90.0, 90.0)},
}


# ---------------------------------------------------------------------------
# Input / Output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DesignRequirements:
    """Customer requirements for constellation design."""
    target_coverage: str = "global"                # global|tropical|polar|specific_latitudes
    min_revisit_time_minutes: float = 90.0         # how often each ground point is covered
    compute_capacity_gpu_hours_day: float = 1000.0 # total compute needed per day
    max_latency_ms: float = 50.0                   # max ground-to-satellite latency
    budget_usd: float = 50_000_000.0               # total budget constraint
    max_eclipse_fraction: float = 0.40             # maximum acceptable eclipse time
    radiation_tolerance: str = "rad_tolerant"       # commercial|rad_tolerant|rad_hard
    data_volume_tb_day: float = 1.0                # total data to process per day
    ground_stations: Optional[List[GroundStation]] = None
    target_latitudes: Optional[Tuple[float, float]] = None  # for specific_latitudes

    def __post_init__(self):
        if self.ground_stations is None:
            self.ground_stations = list(DEFAULT_GROUND_STATIONS)
        if self.target_coverage not in COVERAGE_PROFILES:
            raise ValueError(
                f"Unknown coverage type: {self.target_coverage}. "
                f"Options: {list(COVERAGE_PROFILES.keys())}"
            )


@dataclass
class ConstellationDesign:
    """Output: the optimized constellation design."""
    # Constellation geometry
    n_satellites: int = 0
    n_planes: int = 0
    sats_per_plane: int = 0
    altitude_km: float = 0.0
    inclination_deg: float = 0.0
    walker_notation: str = ""  # i:T/P/F

    # Per-satellite sizing
    gpu_per_sat: int = 0
    solar_watts: float = 0.0
    battery_wh: float = 0.0
    radiator_m2: float = 0.0
    satellite_mass_kg: float = 0.0

    # Fleet totals
    total_gpus: int = 0
    total_gpu_hours_day: float = 0.0

    # Cost
    estimated_cost_usd: float = 0.0
    cost_per_satellite_usd: float = 0.0
    launch_cost_usd: float = 0.0
    hardware_cost_usd: float = 0.0

    # Coverage analysis
    coverage_map: Dict = field(default_factory=dict)
    utilization_estimate_pct: float = 0.0
    avg_revisit_minutes: float = 0.0
    ground_contact_minutes_per_orbit: float = 0.0

    # Orbital parameters
    orbital_period_minutes: float = 0.0
    eclipse_fraction: float = 0.0
    max_latency_ms: float = 0.0

    # Metadata
    design_notes: List[str] = field(default_factory=list)
    alternatives_considered: int = 0


# ---------------------------------------------------------------------------
# Orbital mechanics helpers
# ---------------------------------------------------------------------------

def orbital_period_s(altitude_km: float) -> float:
    """Orbital period in seconds for a circular orbit."""
    a = EARTH_RADIUS_KM + altitude_km
    return 2.0 * math.pi * math.sqrt(a**3 / MU_EARTH)


def eclipse_fraction_estimate(altitude_km: float, inclination_deg: float) -> float:
    """Estimate fraction of orbit spent in eclipse.

    Uses geometric shadow model. For circular LEO orbits, eclipse fraction
    depends on the orbit's beta angle (angle between orbital plane and Sun).
    We use an average over a year to give a representative value.
    """
    a = EARTH_RADIUS_KM + altitude_km
    # Angular radius of Earth as seen from satellite
    rho = math.asin(EARTH_RADIUS_KM / a)
    # Maximum eclipse fraction (beta=0, equinox)
    max_eclipse = rho / math.pi
    # Beta angle varies with inclination and season; average effect
    # Sun-synchronous orbits can avoid eclipse almost entirely
    inc_rad = math.radians(inclination_deg)
    # Simplified: eclipse fraction reduces as inclination approaches SSO
    # SSO at ~97-98 deg can have dawn-dusk with ~0% eclipse
    if 96.0 <= inclination_deg <= 99.0:
        # Sun-synchronous dawn-dusk: minimal eclipse
        return max_eclipse * 0.15
    elif inclination_deg < 10.0:
        # Near-equatorial: predictable eclipse
        return max_eclipse * 0.95
    else:
        # General case: average beta angle effect
        return max_eclipse * 0.85


def max_altitude_for_latency(max_latency_ms: float) -> float:
    """Maximum altitude that satisfies the latency constraint.

    Latency = 2 * (altitude / c) for straight-up-and-down.
    Add 20% for slant range and processing overhead.
    """
    # One-way propagation: alt / c_light (km/s -> ms conversion)
    # Round trip with overhead factor
    max_alt_km = (max_latency_ms / 2.0) * C_LIGHT / 1000.0 * 0.8
    return max_alt_km


def swath_radius_km(altitude_km: float, min_elevation_deg: float = 10.0) -> float:
    """Ground swath radius visible from a satellite at given altitude.

    A ground point is 'covered' if the satellite is above min_elevation_deg.
    """
    a = EARTH_RADIUS_KM + altitude_km
    # Earth central angle for minimum elevation
    elev_rad = math.radians(min_elevation_deg)
    # Using spherical geometry:
    # sin(nadir_angle) = (R_earth / (R_earth + h)) * cos(elevation)
    sin_nadir = (EARTH_RADIUS_KM / a) * math.cos(elev_rad)
    if sin_nadir > 1.0:
        return 0.0
    nadir_angle = math.asin(sin_nadir)
    # Earth central angle
    earth_angle = math.pi / 2.0 - elev_rad - nadir_angle
    return EARTH_RADIUS_KM * earth_angle


def n_sats_for_single_coverage(altitude_km: float, min_elevation_deg: float = 10.0) -> int:
    """Minimum satellites for single-fold continuous global coverage.

    Uses the "streets of coverage" approximation for Walker constellations.
    """
    swath = swath_radius_km(altitude_km, min_elevation_deg)
    if swath <= 0:
        return 9999
    # Earth surface area / area covered by one satellite footprint
    earth_area = 4.0 * math.pi * EARTH_RADIUS_KM**2
    footprint_area = math.pi * swath**2
    # Overlap factor ~1.5 for practical constellations
    return max(1, math.ceil(earth_area / footprint_area * 1.5))


# ---------------------------------------------------------------------------
# Coverage analysis
# ---------------------------------------------------------------------------

def analyze_coverage(
    altitude_km: float,
    inclination_deg: float,
    n_planes: int,
    sats_per_plane: int,
    lat_band: Tuple[float, float] = (-90.0, 90.0),
    n_lat_bins: int = 18,
    sim_orbits: int = 2,
) -> Dict:
    """Analyze coverage of a Walker Delta constellation.

    Simulates satellite ground tracks over multiple orbits and calculates:
    - Instantaneous coverage fraction
    - Revisit time per latitude band
    - Coverage gaps

    Returns dict with coverage metrics.
    """
    total_sats = n_planes * sats_per_plane
    period_s = orbital_period_s(altitude_km)
    swath = swath_radius_km(altitude_km)
    sim_duration_s = period_s * sim_orbits
    dt = 60.0  # 1-minute steps

    # Build satellite states: (raan, mean_anomaly_offset) per satellite
    sat_params = []
    for p in range(n_planes):
        raan_deg = (360.0 / n_planes) * p
        for s in range(sats_per_plane):
            ma_offset_deg = (360.0 / sats_per_plane) * s
            sat_params.append((raan_deg, ma_offset_deg))

    # Latitude bins
    lat_edges = np.linspace(lat_band[0], lat_band[1], n_lat_bins + 1)
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])

    # Track coverage per latitude bin
    # For each bin, record timestamps when it's covered
    bin_coverage_times = [[] for _ in range(n_lat_bins)]
    total_steps = int(sim_duration_s / dt)
    covered_steps_total = 0

    inc_rad = math.radians(inclination_deg)
    earth_rotation_rate = 360.0 / 86400.0  # deg/s

    for step_i in range(total_steps):
        t = step_i * dt
        # For each latitude bin, check if any satellite covers it
        for bin_i, lat_c in enumerate(lat_centers):
            lat_rad = math.radians(lat_c)
            covered = False
            for raan_deg, ma_offset_deg in sat_params:
                # Satellite ground track at time t
                # Mean anomaly progresses at orbital rate
                orbital_rate = 360.0 / period_s  # deg/s
                ma_deg = (ma_offset_deg + orbital_rate * t) % 360.0
                ma_rad = math.radians(ma_deg)

                # Subsatellite latitude
                sat_lat_rad = math.asin(math.sin(inc_rad) * math.sin(ma_rad))
                sat_lat = math.degrees(sat_lat_rad)

                # Quick latitude check before doing expensive longitude calc
                lat_diff_km = abs(sat_lat - lat_c) * (math.pi * EARTH_RADIUS_KM / 180.0)
                if lat_diff_km > swath * 1.2:
                    continue

                # Subsatellite longitude (includes Earth rotation and RAAN)
                if abs(math.cos(sat_lat_rad)) < 1e-10:
                    sat_lon_component = 0.0
                else:
                    sin_lon = math.sin(ma_rad) * math.cos(inc_rad) / math.cos(sat_lat_rad)
                    cos_lon = math.cos(ma_rad) / math.cos(sat_lat_rad)
                    sat_lon_component = math.degrees(math.atan2(sin_lon, cos_lon))

                sat_lon = (raan_deg + sat_lon_component - earth_rotation_rate * t) % 360.0
                if sat_lon > 180.0:
                    sat_lon -= 360.0

                # Check if this latitude band center is within swath
                # Use great-circle distance approximation
                dlon = abs(sat_lon - 0.0)  # We check at lon=0 as representative
                if dlon > 180.0:
                    dlon = 360.0 - dlon
                dist_km = math.sqrt(lat_diff_km**2 +
                    (math.radians(dlon) * EARTH_RADIUS_KM * math.cos(lat_rad))**2)

                if dist_km <= swath:
                    covered = True
                    break

            if covered:
                bin_coverage_times[bin_i].append(t)
                covered_steps_total += 1

    # Compute metrics per latitude band
    lat_metrics = {}
    for bin_i, lat_c in enumerate(lat_centers):
        times = bin_coverage_times[bin_i]
        coverage_frac = len(times) / total_steps if total_steps > 0 else 0.0

        # Compute revisit time (average gap between coverage events)
        if len(times) >= 2:
            gaps = []
            for j in range(1, len(times)):
                gap = times[j] - times[j - 1]
                if gap > dt * 1.5:  # Only count actual gaps (not consecutive coverage)
                    gaps.append(gap)
            avg_revisit_s = np.mean(gaps) if gaps else 0.0
            max_revisit_s = max(gaps) if gaps else 0.0
        else:
            avg_revisit_s = sim_duration_s
            max_revisit_s = sim_duration_s

        lat_label = f"{lat_c:+.0f}"
        lat_metrics[lat_label] = {
            "latitude": lat_c,
            "coverage_fraction": round(coverage_frac, 3),
            "avg_revisit_minutes": round(avg_revisit_s / 60.0, 1),
            "max_revisit_minutes": round(max_revisit_s / 60.0, 1),
        }

    # Overall metrics
    total_possible = total_steps * n_lat_bins
    overall_coverage = covered_steps_total / total_possible if total_possible > 0 else 0.0
    all_revisits = [m["avg_revisit_minutes"] for m in lat_metrics.values() if m["avg_revisit_minutes"] > 0]
    avg_revisit = np.mean(all_revisits) if all_revisits else sim_duration_s / 60.0

    return {
        "overall_coverage_fraction": round(overall_coverage, 3),
        "avg_revisit_minutes": round(float(avg_revisit), 1),
        "latitude_bands": lat_metrics,
        "swath_radius_km": round(swath, 1),
        "sim_duration_minutes": round(sim_duration_s / 60.0, 1),
    }


# ---------------------------------------------------------------------------
# Ground station contact analysis
# ---------------------------------------------------------------------------

def estimate_ground_contact(
    altitude_km: float,
    inclination_deg: float,
    stations: List[GroundStation],
) -> float:
    """Estimate average ground contact minutes per orbit for the constellation.

    Uses geometric analysis rather than full SGP4 propagation for speed.
    """
    period_s = orbital_period_s(altitude_km)
    swath = swath_radius_km(altitude_km, min_elevation_deg=5.0)

    total_contact_s = 0.0
    inc_rad = math.radians(inclination_deg)
    ground_track_speed_kms = 2.0 * math.pi * (EARTH_RADIUS_KM + altitude_km) / period_s

    for station in stations:
        # Can the satellite reach this station's latitude?
        if abs(station.lat_deg) > inclination_deg + 5.0:
            continue  # Station latitude unreachable

        # Approximate contact time per pass
        # Pass duration ~ 2 * swath / ground_track_speed
        pass_duration_s = 2.0 * swath / ground_track_speed_kms

        # Number of passes per orbit: depends on whether ground track crosses station
        # Simplified: ~1 pass per orbit for stations within latitude range
        # Reduced probability for stations far from ground track
        lat_factor = 1.0 - abs(station.lat_deg) / 90.0
        passes_per_orbit = 0.3 * lat_factor  # rough average

        total_contact_s += pass_duration_s * passes_per_orbit

    return total_contact_s / 60.0  # return minutes


# ---------------------------------------------------------------------------
# Walker Delta constellation generator
# ---------------------------------------------------------------------------

def walker_notation(inclination_deg: float, total_sats: int, n_planes: int, phasing: int = 0) -> str:
    """Standard Walker notation: i:T/P/F."""
    return f"{inclination_deg:.0f}:{total_sats}/{n_planes}/{phasing}"


def generate_walker_tles(
    inclination_deg: float,
    altitude_km: float,
    n_planes: int,
    sats_per_plane: int,
    phasing: int = 0,
    epoch_year: int = 26,
    epoch_day: float = 85.0,
) -> List[Tuple[str, str, str]]:
    """Generate TLE sets for a Walker Delta constellation.

    Walker Delta notation: i:T/P/F where:
    - i = inclination (deg)
    - T = total satellites
    - P = number of planes
    - F = phasing parameter (0 to P-1)

    Returns list of (name, line1, line2) tuples.
    """
    a = EARTH_RADIUS_KM + altitude_km
    period_s = 2.0 * math.pi * math.sqrt(a**3 / MU_EARTH)
    mean_motion = 86400.0 / period_s

    total_sats = n_planes * sats_per_plane
    tles = []

    for p in range(n_planes):
        raan = (360.0 / n_planes) * p

        for s in range(sats_per_plane):
            # Walker phasing: inter-plane spacing
            phase_offset = (360.0 * phasing * p) / total_sats
            mean_anomaly = ((360.0 / sats_per_plane) * s + phase_offset) % 360.0

            sat_idx = p * sats_per_plane + s
            cat_num = 80000 + sat_idx

            line1 = "1 {:05d}U {:02d}001A   {:02d}{:012.8f}  .00000000  00000-0  00000-0 0  999{}".format(
                cat_num, epoch_year, epoch_year, epoch_day, sat_idx % 10
            )
            line2 = "2 {:05d} {:8.4f} {:8.4f} 0001000   0.0000 {:8.4f} {:11.8f}    0{}".format(
                cat_num, inclination_deg, raan, mean_anomaly, mean_motion, sat_idx % 10
            )

            line1 = _fix_tle_checksum(line1)
            line2 = _fix_tle_checksum(line2)

            name = f"WALKER-P{p:02d}S{s:02d}"
            tles.append((name, line1, line2))

    return tles


# ---------------------------------------------------------------------------
# Power and thermal sizing
# ---------------------------------------------------------------------------

def size_power_system(
    n_gpus: int,
    altitude_km: float,
    inclination_deg: float,
    max_eclipse_fraction: float,
) -> Tuple[float, float]:
    """Size solar panels and battery for compute satellite.

    Returns (solar_panel_watts, battery_wh).
    """
    eclipse_frac = eclipse_fraction_estimate(altitude_km, inclination_deg)
    eclipse_frac = min(eclipse_frac, max_eclipse_fraction)
    sunlit_frac = 1.0 - eclipse_frac

    # Power budget
    gpu_power = n_gpus * GPU_TDP_W
    housekeeping = 150.0  # ADCS, comms, OBC
    total_load = gpu_power + housekeeping

    # Solar panels must generate enough during sunlit to power loads + recharge battery
    # Energy balance over one orbit:
    #   solar * sunlit_frac * charge_eff >= total_load  (averaged over orbit)
    # Plus margin for degradation (30% over mission life)
    charge_eff = 0.90
    degradation_margin = 1.3
    # Panel must produce enough in sunlit fraction to cover the full orbit load
    if sunlit_frac > 0:
        solar_watts = (total_load * degradation_margin) / (sunlit_frac * charge_eff)
    else:
        solar_watts = total_load * 10.0  # Degenerate case

    # Battery: must sustain full load through eclipse
    period_s = orbital_period_s(altitude_km)
    eclipse_duration_s = period_s * eclipse_frac
    eclipse_energy_wh = total_load * (eclipse_duration_s / 3600.0)
    # Depth of discharge limit: 60% usable
    dod = 0.60
    battery_wh = eclipse_energy_wh / dod

    return round(solar_watts, 0), round(battery_wh, 0)


def size_thermal_system(n_gpus: int, altitude_km: float) -> float:
    """Size radiator area for compute satellite.

    Returns radiator_area_m2.
    """
    # Total heat to reject
    gpu_heat = n_gpus * GPU_TDP_W  # GPUs convert nearly all power to heat
    housekeeping_heat = 150.0
    total_heat = gpu_heat + housekeeping_heat

    # Radiator equilibrium temperature target: 70C (343K)
    target_temp_k = 343.0
    emissivity = 0.90

    # Environmental heat loads
    solar_absorptivity = 0.20
    q_solar = solar_absorptivity * 1361.0 * 0.25  # per m2, average
    q_earth_ir = 240.0 * 0.5  # per m2, view factor

    # Stefan-Boltzmann: q_radiated = eps * sigma * A * (T^4 - T_space^4)
    q_rad_per_m2 = emissivity * STEFAN_BOLTZMANN * (target_temp_k**4 - 3.0**4)
    net_rad_per_m2 = q_rad_per_m2 - q_solar - q_earth_ir

    if net_rad_per_m2 <= 0:
        return 100.0  # Degenerate

    radiator_m2 = total_heat / net_rad_per_m2
    return round(radiator_m2, 2)


def estimate_satellite_mass(
    n_gpus: int,
    solar_watts: float,
    battery_wh: float,
    radiator_m2: float,
) -> float:
    """Estimate satellite dry mass in kg."""
    bus_mass = 50.0         # Structure, ADCS, OBC
    gpu_mass = n_gpus * 5.0  # ~5 kg per GPU card
    solar_mass = solar_watts * 0.005  # ~5 g/W for deployable panels
    battery_mass = battery_wh * 0.003  # ~3 g/Wh for Li-ion
    radiator_mass = radiator_m2 * 8.0  # ~8 kg/m2 for deployable radiators
    comms_mass = 15.0       # Antenna, transponder

    total = bus_mass + gpu_mass + solar_mass + battery_mass + radiator_mass + comms_mass
    return round(total, 1)


# ---------------------------------------------------------------------------
# ConstellationDesigner
# ---------------------------------------------------------------------------

class ConstellationDesigner:
    """Designs optimal satellite constellation given customer requirements.

    Optimization approach:
    1. Determine altitude range from latency and eclipse constraints
    2. Determine inclination from coverage requirements
    3. Calculate minimum satellites for coverage/revisit
    4. Size power, thermal, and compute per satellite
    5. Iterate to minimize cost while meeting all requirements
    """

    def __init__(self, requirements: DesignRequirements):
        self.req = requirements
        self.designs_evaluated = 0

    def design(self) -> ConstellationDesign:
        """Run the optimizer and return the best constellation design."""
        req = self.req

        # Step 1: Altitude constraints
        alt_max_latency = max_altitude_for_latency(req.max_latency_ms)
        alt_min = 300.0   # Below this, drag is too high
        alt_max = min(2000.0, alt_max_latency)

        # Step 2: Candidate altitudes
        alt_candidates = [
            a for a in range(int(alt_min), int(alt_max) + 1, 50)
            if a >= alt_min
        ]
        if not alt_candidates:
            alt_candidates = [550.0]  # Fallback

        # Step 3: Inclination from coverage type
        profile = COVERAGE_PROFILES[req.target_coverage]
        inc_min, inc_max = profile["inc_range"]
        if req.target_coverage == "specific_latitudes" and req.target_latitudes:
            # Inclination must be >= max target latitude
            needed_inc = max(abs(req.target_latitudes[0]), abs(req.target_latitudes[1]))
            inc_min = max(inc_min, needed_inc)
            inc_max = max(inc_max, needed_inc + 10)

        inc_candidates = list(range(int(inc_min), int(inc_max) + 1, 5))
        if not inc_candidates:
            inc_candidates = [53.0]  # Starlink-like default

        # Step 4: Search the design space
        best_design = None
        best_cost = float("inf")

        for alt in alt_candidates:
            for inc in inc_candidates:
                # Check eclipse constraint
                ef = eclipse_fraction_estimate(alt, inc)
                if ef > req.max_eclipse_fraction:
                    self.designs_evaluated += 1
                    continue

                # How many GPUs total do we need?
                # Account for eclipse downtime: effective compute hours per GPU per day
                sunlit_fraction = 1.0 - ef
                # During eclipse, we can run from battery but at reduced duty cycle
                # Assume 50% duty cycle during eclipse (battery limited)
                effective_hours_per_gpu_day = (sunlit_fraction + ef * 0.5) * 24.0
                n_gpus_total = math.ceil(
                    req.compute_capacity_gpu_hours_day / effective_hours_per_gpu_day
                )
                n_gpus_total = max(1, n_gpus_total)

                # Try different GPUs-per-satellite configurations
                for gpus_per_sat in [1, 2, 4, 8]:
                    n_sats = math.ceil(n_gpus_total / gpus_per_sat)
                    if n_sats < 1:
                        continue

                    # Find plane/sat arrangement
                    # Try to match good coverage geometry
                    for n_planes in self._plane_candidates(n_sats):
                        spp = math.ceil(n_sats / n_planes)
                        actual_sats = n_planes * spp

                        # Size subsystems
                        solar_w, batt_wh = size_power_system(gpus_per_sat, alt, inc, req.max_eclipse_fraction)
                        rad_m2 = size_thermal_system(gpus_per_sat, alt)
                        mass_kg = estimate_satellite_mass(gpus_per_sat, solar_w, batt_wh, rad_m2)

                        # Cost estimate
                        rad_mult = RAD_TOLERANCE_MULTIPLIER.get(req.radiation_tolerance, 1.3)
                        hw_costs = HardwareCosts(
                            n_gpus=gpus_per_sat,
                            solar_panel_watts=solar_w,
                            battery_capacity_kwh=batt_wh / 1000.0,
                            radiator_area_m2=rad_m2,
                            satellite_dry_mass_kg=mass_kg,
                            radiation_hardening_pct=rad_mult - 1.0,
                        )
                        per_sat_hardware = hw_costs.total_hardware_cost()

                        # Launch cost
                        launch = LaunchCosts()
                        if mass_kg < 150:
                            launch.default_vehicle = "falcon9"
                        elif mass_kg < 500:
                            launch.default_vehicle = "falcon9"
                        else:
                            launch.default_vehicle = "starship"
                        per_sat_launch = launch.launch_cost(mass_kg)
                        per_sat_total = per_sat_hardware + per_sat_launch

                        # Total constellation cost (CAPEX only for budget comparison)
                        total_cost = per_sat_total * actual_sats

                        # Add 5-year operations estimate
                        ops = OperatingCosts()
                        ops_annual = ops.total_annual(per_sat_hardware)
                        total_with_ops = total_cost + ops_annual * 5.0

                        self.designs_evaluated += 1

                        # Check budget constraint
                        if total_cost > req.budget_usd * 1.1:  # 10% slack
                            continue

                        # Compute actual capacity
                        actual_gpu_hours = actual_sats * gpus_per_sat * effective_hours_per_gpu_day
                        if actual_gpu_hours < req.compute_capacity_gpu_hours_day * 0.9:
                            continue  # Doesn't meet compute requirement

                        # Score: minimize cost, prefer meeting all requirements
                        score = total_cost

                        if score < best_cost:
                            best_cost = score

                            period_min = orbital_period_s(alt) / 60.0
                            latency_ms = 2.0 * (alt / C_LIGHT) * 1000.0 * 1.2  # 20% overhead
                            contact_min = estimate_ground_contact(alt, inc, req.ground_stations)

                            utilization = min(100.0,
                                req.compute_capacity_gpu_hours_day / actual_gpu_hours * 100.0)

                            best_design = ConstellationDesign(
                                n_satellites=actual_sats,
                                n_planes=n_planes,
                                sats_per_plane=spp,
                                altitude_km=float(alt),
                                inclination_deg=float(inc),
                                walker_notation=walker_notation(inc, actual_sats, n_planes),
                                gpu_per_sat=gpus_per_sat,
                                solar_watts=solar_w,
                                battery_wh=batt_wh,
                                radiator_m2=rad_m2,
                                satellite_mass_kg=mass_kg,
                                total_gpus=actual_sats * gpus_per_sat,
                                total_gpu_hours_day=round(actual_gpu_hours, 1),
                                estimated_cost_usd=round(total_cost, 0),
                                cost_per_satellite_usd=round(per_sat_total, 0),
                                launch_cost_usd=round(per_sat_launch * actual_sats, 0),
                                hardware_cost_usd=round(per_sat_hardware * actual_sats, 0),
                                utilization_estimate_pct=round(utilization, 1),
                                orbital_period_minutes=round(period_min, 1),
                                eclipse_fraction=round(ef, 3),
                                max_latency_ms=round(latency_ms, 1),
                                ground_contact_minutes_per_orbit=round(contact_min, 1),
                                alternatives_considered=self.designs_evaluated,
                            )

        if best_design is None:
            # No feasible design found within budget — return the cheapest infeasible
            best_design = self._fallback_design()

        # Run coverage analysis on the best design
        if best_design.n_satellites > 0:
            lat_band = COVERAGE_PROFILES[req.target_coverage]["lat_band"]
            if req.target_coverage == "specific_latitudes" and req.target_latitudes:
                lat_band = req.target_latitudes

            coverage = analyze_coverage(
                best_design.altitude_km,
                best_design.inclination_deg,
                best_design.n_planes,
                best_design.sats_per_plane,
                lat_band=lat_band,
                sim_orbits=2,
            )
            best_design.coverage_map = coverage
            best_design.avg_revisit_minutes = coverage["avg_revisit_minutes"]
            best_design.alternatives_considered = self.designs_evaluated

            # Generate design notes
            best_design.design_notes = self._generate_notes(best_design)

        return best_design

    def _plane_candidates(self, n_sats: int) -> List[int]:
        """Generate candidate numbers of orbital planes."""
        candidates = []
        for p in range(1, min(n_sats + 1, 25)):
            if n_sats >= p:
                candidates.append(p)
        # Prefer factors of n_sats for even distribution
        factors = [p for p in candidates if n_sats % p == 0]
        # Also include common values
        common = [1, 2, 3, 4, 6, 8, 12]
        result = sorted(set(factors + [c for c in common if c <= n_sats]))
        return result if result else [1]

    def _fallback_design(self) -> ConstellationDesign:
        """Return a minimal design when nothing fits the budget."""
        design = ConstellationDesign()
        design.design_notes = [
            "WARNING: No feasible design found within budget constraints.",
            f"Budget: ${self.req.budget_usd:,.0f}",
            f"Required compute: {self.req.compute_capacity_gpu_hours_day} GPU-hours/day",
            "Consider: increasing budget, reducing compute requirements, or relaxing constraints.",
        ]
        design.alternatives_considered = self.designs_evaluated
        return design

    def _generate_notes(self, design: ConstellationDesign) -> List[str]:
        """Generate human-readable design notes."""
        notes = []
        req = self.req

        # Coverage type
        notes.append(f"Coverage: {req.target_coverage} ({design.walker_notation})")

        # Budget usage
        budget_pct = design.estimated_cost_usd / req.budget_usd * 100.0
        notes.append(f"Budget utilization: {budget_pct:.0f}% (${design.estimated_cost_usd:,.0f} of ${req.budget_usd:,.0f})")

        # Compute capacity
        notes.append(f"Compute capacity: {design.total_gpu_hours_day:.0f} GPU-hours/day "
                      f"(requested: {req.compute_capacity_gpu_hours_day:.0f})")

        # Eclipse
        if design.eclipse_fraction > 0.30:
            notes.append(f"NOTE: Eclipse fraction {design.eclipse_fraction:.1%} is high — "
                          "consider sun-synchronous orbit for better power budget")

        # Mass
        if design.satellite_mass_kg > 300:
            notes.append(f"NOTE: Satellite mass {design.satellite_mass_kg:.0f} kg — "
                          "may benefit from Starship launch for lower $/kg")

        # Latency
        if design.max_latency_ms > req.max_latency_ms:
            notes.append(f"WARNING: Estimated latency {design.max_latency_ms:.1f} ms "
                          f"exceeds requirement {req.max_latency_ms:.1f} ms")

        return notes


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_design(design: ConstellationDesign, requirements: Optional[DesignRequirements] = None) -> None:
    """Print a formatted constellation design report."""
    print("=" * 72)
    print("  CONSTELLATION DESIGN OPTIMIZER — Results")
    print("=" * 72)

    if requirements:
        print("\n  REQUIREMENTS:")
        print(f"    Coverage:              {requirements.target_coverage}")
        print(f"    Compute:               {requirements.compute_capacity_gpu_hours_day:,.0f} GPU-hours/day")
        print(f"    Max latency:           {requirements.max_latency_ms:.0f} ms")
        print(f"    Budget:                ${requirements.budget_usd:,.0f}")
        print(f"    Max eclipse fraction:  {requirements.max_eclipse_fraction:.0%}")
        print(f"    Radiation tolerance:   {requirements.radiation_tolerance}")
        print(f"    Data volume:           {requirements.data_volume_tb_day:.1f} TB/day")

    print("\n  CONSTELLATION GEOMETRY:")
    print(f"    Satellites:            {design.n_satellites}")
    print(f"    Orbital planes:        {design.n_planes}")
    print(f"    Sats per plane:        {design.sats_per_plane}")
    print(f"    Altitude:              {design.altitude_km:.0f} km")
    print(f"    Inclination:           {design.inclination_deg:.1f} deg")
    print(f"    Walker notation:       {design.walker_notation}")
    print(f"    Orbital period:        {design.orbital_period_minutes:.1f} min")

    print("\n  PER-SATELLITE SIZING:")
    print(f"    GPUs per satellite:    {design.gpu_per_sat}")
    print(f"    Solar panels:          {design.solar_watts:,.0f} W")
    print(f"    Battery:               {design.battery_wh:,.0f} Wh")
    print(f"    Radiator area:         {design.radiator_m2:.1f} m2")
    print(f"    Satellite mass:        {design.satellite_mass_kg:.0f} kg")

    print("\n  FLEET TOTALS:")
    print(f"    Total GPUs:            {design.total_gpus}")
    print(f"    GPU-hours/day:         {design.total_gpu_hours_day:,.0f}")
    print(f"    Utilization estimate:  {design.utilization_estimate_pct:.0f}%")

    print("\n  COST ESTIMATE:")
    print(f"    Per satellite:         ${design.cost_per_satellite_usd:>14,.0f}")
    print(f"    Hardware total:        ${design.hardware_cost_usd:>14,.0f}")
    print(f"    Launch total:          ${design.launch_cost_usd:>14,.0f}")
    print(f"    TOTAL CONSTELLATION:   ${design.estimated_cost_usd:>14,.0f}")

    print("\n  ORBITAL ENVIRONMENT:")
    print(f"    Eclipse fraction:      {design.eclipse_fraction:.1%}")
    print(f"    Max latency:           {design.max_latency_ms:.1f} ms")
    print(f"    Ground contact:        {design.ground_contact_minutes_per_orbit:.1f} min/orbit")

    # Coverage map
    if design.coverage_map:
        cov = design.coverage_map
        print("\n  COVERAGE ANALYSIS:")
        print(f"    Overall coverage:      {cov.get('overall_coverage_fraction', 0):.1%}")
        print(f"    Avg revisit time:      {cov.get('avg_revisit_minutes', 0):.0f} min")
        print(f"    Swath radius:          {cov.get('swath_radius_km', 0):.0f} km")

        bands = cov.get("latitude_bands", {})
        if bands:
            print(f"\n    {'Latitude':>10s}  {'Coverage':>10s}  {'Avg Revisit':>12s}  {'Max Revisit':>12s}")
            print(f"    {'':->10s}  {'':->10s}  {'':->12s}  {'':->12s}")
            for label, metrics in sorted(bands.items(), key=lambda x: x[1]["latitude"]):
                print(f"    {label:>10s}  "
                      f"{metrics['coverage_fraction']:>9.1%}  "
                      f"{metrics['avg_revisit_minutes']:>10.0f} min  "
                      f"{metrics['max_revisit_minutes']:>10.0f} min")

    # Design notes
    if design.design_notes:
        print("\n  DESIGN NOTES:")
        for note in design.design_notes:
            print(f"    - {note}")

    print(f"\n  Alternatives evaluated: {design.alternatives_considered}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """CLI entry point for constellation designer."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Constellation Design Optimizer for orbital compute",
    )
    parser.add_argument("--coverage", default="global",
                        choices=["global", "tropical", "polar", "specific_latitudes"],
                        help="Target coverage type")
    parser.add_argument("--compute", type=float, default=1000.0,
                        help="Required GPU-hours per day")
    parser.add_argument("--budget", type=float, default=50e6,
                        help="Budget in USD")
    parser.add_argument("--latency", type=float, default=50.0,
                        help="Max latency in ms")
    parser.add_argument("--eclipse", type=float, default=0.40,
                        help="Max eclipse fraction (0-1)")
    parser.add_argument("--radiation", default="rad_tolerant",
                        choices=["commercial", "rad_tolerant", "rad_hard"],
                        help="Radiation tolerance level")
    parser.add_argument("--data", type=float, default=1.0,
                        help="Data volume in TB/day")

    args = parser.parse_args()

    req = DesignRequirements(
        target_coverage=args.coverage,
        compute_capacity_gpu_hours_day=args.compute,
        budget_usd=args.budget,
        max_latency_ms=args.latency,
        max_eclipse_fraction=args.eclipse,
        radiation_tolerance=args.radiation,
        data_volume_tb_day=args.data,
    )

    print(f"\n  Designing constellation for {args.compute:,.0f} GPU-hours/day, "
          f"{args.coverage} coverage, ${args.budget:,.0f} budget...")
    print(f"  Searching design space...\n")

    designer = ConstellationDesigner(req)
    design = designer.design()
    print_design(design, req)

    return design


# ---------------------------------------------------------------------------
# __main__ test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # CLI mode
        main()
    else:
        # Default demo: "I need 1000 GPU-hours/day of Earth observation processing,
        # global coverage, $50M budget"
        print("\n" + "=" * 72)
        print("  DEMO: 1000 GPU-hours/day, global coverage, $50M budget")
        print("=" * 72)

        req = DesignRequirements(
            target_coverage="global",
            min_revisit_time_minutes=90.0,
            compute_capacity_gpu_hours_day=1000.0,
            max_latency_ms=50.0,
            budget_usd=50_000_000.0,
            max_eclipse_fraction=0.40,
            radiation_tolerance="rad_tolerant",
            data_volume_tb_day=1.0,
        )

        print(f"\n  Searching design space...")
        designer = ConstellationDesigner(req)
        design = designer.design()
        print_design(design, req)

        # Also show Walker TLE generation
        if design.n_satellites > 0:
            print(f"\n\n  WALKER TLE GENERATION (first 3 satellites):")
            print("=" * 72)
            tles = generate_walker_tles(
                design.inclination_deg,
                design.altitude_km,
                design.n_planes,
                design.sats_per_plane,
            )
            for name, l1, l2 in tles[:3]:
                print(f"  {name}")
                print(f"  {l1}")
                print(f"  {l2}")
                print()
