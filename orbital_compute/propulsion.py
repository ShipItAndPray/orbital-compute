"""Propulsion and orbit maintenance — station-keeping, deorbit, and compute impact.

Models atmospheric drag, propulsion systems, and their effect on compute operations.
Satellite compute constellations need station-keeping to maintain orbit altitude,
and this module calculates the propellant budget, maneuver schedule, and compute
downtime from thruster firings.

Reference atmosphere model: US Standard Atmosphere 1976 (simplified exponential).
Drag coefficients: typical for flat-plate LEO satellites (Cd ~ 2.2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EARTH_RADIUS_KM = 6371.0
EARTH_MU_KM3S2 = 398600.4418       # Gravitational parameter (km^3/s^2)
G0_MS2 = 9.80665                    # Standard gravity (m/s^2)

# Exponential atmosphere reference values (US Standard Atmosphere 1976, simplified)
# Each band: (base_altitude_km, base_density_kg_m3, scale_height_km)
_ATMO_BANDS: List[Tuple[float, float, float]] = [
    (200.0, 2.53e-10, 37.0),
    (300.0, 7.22e-11, 43.0),
    (400.0, 2.80e-11, 53.0),
    (500.0, 8.48e-12, 63.0),
    (600.0, 2.14e-12, 71.0),
    (700.0, 3.83e-13, 81.0),
    (800.0, 7.21e-14, 89.0),
    (900.0, 1.56e-14, 97.0),
    (1000.0, 3.56e-15, 105.0),
    (1200.0, 2.08e-16, 120.0),
]


# ---------------------------------------------------------------------------
# Atmosphere
# ---------------------------------------------------------------------------

@dataclass
class AtmosphericConditions:
    """Atmospheric density at a given altitude."""
    altitude_km: float
    density_kg_m3: float
    scale_height_km: float
    solar_multiplier: float


class SolarActivity(Enum):
    """Solar cycle activity level — affects upper-atmosphere density."""
    LOW = "low"          # Solar minimum
    MODERATE = "moderate"  # Average conditions
    HIGH = "high"        # Solar maximum


# Solar activity multipliers on atmospheric density.
# At solar max the thermosphere can be 5-10x denser than solar min.
_SOLAR_MULTIPLIERS = {
    SolarActivity.LOW: 0.5,
    SolarActivity.MODERATE: 1.0,
    SolarActivity.HIGH: 3.0,
}


def atmospheric_density(altitude_km: float,
                        solar: SolarActivity = SolarActivity.MODERATE) -> AtmosphericConditions:
    """Compute atmospheric density using a piecewise-exponential model.

    Interpolates between reference bands from the US Standard Atmosphere (1976),
    then scales by solar activity.  Below 200 km or above 1500 km the model
    extrapolates (less accurate, but those altitudes are rarely used for
    constellation operations).

    Returns an AtmosphericConditions dataclass.
    """
    # Find the bounding band
    band = _ATMO_BANDS[0]
    for b in _ATMO_BANDS:
        if altitude_km >= b[0]:
            band = b
        else:
            break

    h0, rho0, H = band
    solar_mult = _SOLAR_MULTIPLIERS[solar]
    rho = rho0 * math.exp(-(altitude_km - h0) / H) * solar_mult

    return AtmosphericConditions(
        altitude_km=altitude_km,
        density_kg_m3=rho,
        scale_height_km=H,
        solar_multiplier=solar_mult,
    )


# ---------------------------------------------------------------------------
# Drag and orbit decay
# ---------------------------------------------------------------------------

@dataclass
class DragParameters:
    """Satellite physical properties relevant to drag."""
    mass_kg: float = 260.0         # Satellite mass (Starlink v1.5 ~ 260 kg)
    cross_section_m2: float = 4.0  # Ram-facing cross-section area (m^2)
    drag_coefficient: float = 2.2  # Typical Cd for LEO flat-plate


@dataclass
class DragResult:
    """Drag force and resulting orbit decay rate."""
    drag_force_n: float
    drag_acceleration_ms2: float
    altitude_decay_km_per_day: float
    days_to_deorbit: float          # Without station-keeping


def compute_drag(altitude_km: float,
                 params: DragParameters,
                 solar: SolarActivity = SolarActivity.MODERATE) -> DragResult:
    """Compute drag force, deceleration, and orbit decay rate.

    Uses circular orbit velocity at the given altitude and the exponential
    atmosphere model.  Orbit decay rate is a first-order estimate:
        da/dt ~ -rho * v * (Cd * A / m) * a
    where a is the semi-major axis.
    """
    r_km = EARTH_RADIUS_KM + altitude_km
    r_m = r_km * 1000.0

    # Circular orbit velocity
    v_ms = math.sqrt(EARTH_MU_KM3S2 * 1e9 / r_m)  # km^3/s^2 -> m^3/s^2

    atmo = atmospheric_density(altitude_km, solar)
    rho = atmo.density_kg_m3

    # Ballistic coefficient components
    Cd_A = params.drag_coefficient * params.cross_section_m2
    bc = Cd_A / params.mass_kg  # m^2/kg

    # Drag force: F = 0.5 * rho * v^2 * Cd * A
    drag_force = 0.5 * rho * v_ms ** 2 * Cd_A

    # Drag acceleration
    drag_accel = drag_force / params.mass_kg

    # Semi-major axis decay rate (m/s) -> convert to km/day
    # da/dt = -rho * v * Cd * A / m * a  (approximate for circular orbit)
    da_dt_ms = -rho * v_ms * Cd_A / params.mass_kg * r_m
    altitude_decay_km_day = abs(da_dt_ms) * 86400.0 / 1000.0

    # Time to deorbit (very rough — assumes constant decay rate, which
    # underestimates actual time since drag increases as altitude drops)
    safe_altitude = max(altitude_km - 150.0, 1.0)  # Deorbit ~ 150 km altitude
    if altitude_decay_km_day > 0:
        days_to_deorbit = safe_altitude / altitude_decay_km_day
    else:
        days_to_deorbit = float("inf")

    return DragResult(
        drag_force_n=drag_force,
        drag_acceleration_ms2=drag_accel,
        altitude_decay_km_per_day=altitude_decay_km_day,
        days_to_deorbit=days_to_deorbit,
    )


# ---------------------------------------------------------------------------
# Propulsion systems
# ---------------------------------------------------------------------------

class PropulsionType(Enum):
    """Thruster technology."""
    HALL_THRUSTER = "hall_thruster"
    HYDRAZINE = "hydrazine"
    COLD_GAS = "cold_gas"


@dataclass
class PropulsionSystem:
    """Propulsion system parameters."""
    name: str
    prop_type: PropulsionType
    specific_impulse_s: float   # Isp (seconds)
    thrust_n: float             # Thrust (Newtons)
    power_draw_w: float         # Electrical power required (W)
    dry_mass_kg: float          # Thruster hardware mass (kg)

    @property
    def exhaust_velocity_ms(self) -> float:
        """Effective exhaust velocity (m/s)."""
        return self.specific_impulse_s * G0_MS2

    def propellant_mass_kg(self, delta_v_ms: float, spacecraft_mass_kg: float) -> float:
        """Propellant mass via Tsiolkovsky rocket equation.

        m_prop = m_spacecraft * (exp(dv / ve) - 1)
        """
        ve = self.exhaust_velocity_ms
        return spacecraft_mass_kg * (math.exp(delta_v_ms / ve) - 1.0)

    def burn_duration_s(self, delta_v_ms: float, spacecraft_mass_kg: float) -> float:
        """How long (seconds) the thruster must fire for the given delta-v.

        For constant thrust: dt = m * dv / F  (approximate — ignores mass change).
        """
        return spacecraft_mass_kg * delta_v_ms / self.thrust_n


# Pre-built propulsion systems
HALL_THRUSTER = PropulsionSystem(
    name="SPT-100 Hall Thruster",
    prop_type=PropulsionType.HALL_THRUSTER,
    specific_impulse_s=1500.0,
    thrust_n=0.083,          # 83 mN
    power_draw_w=1350.0,
    dry_mass_kg=5.0,
)

HYDRAZINE = PropulsionSystem(
    name="MR-103 Hydrazine Thruster",
    prop_type=PropulsionType.HYDRAZINE,
    specific_impulse_s=220.0,
    thrust_n=1.1,            # 1.1 N
    power_draw_w=15.0,
    dry_mass_kg=0.33,
)

COLD_GAS = PropulsionSystem(
    name="Nitrogen Cold Gas Thruster",
    prop_type=PropulsionType.COLD_GAS,
    specific_impulse_s=70.0,
    thrust_n=0.5,
    power_draw_w=5.0,
    dry_mass_kg=0.5,
)


# ---------------------------------------------------------------------------
# Station-keeping budget
# ---------------------------------------------------------------------------

@dataclass
class StationKeepingBudget:
    """Annual station-keeping propulsion budget."""
    altitude_km: float
    delta_v_per_year_ms: float
    propellant_per_year_kg: float
    burn_time_per_year_hours: float
    maneuvers_per_year: int
    propulsion_system: str


def station_keeping_delta_v(altitude_km: float,
                            solar: SolarActivity = SolarActivity.MODERATE,
                            drag_params: Optional[DragParameters] = None) -> float:
    """Compute annual delta-V (m/s) needed to maintain altitude.

    Strategy: counteract drag-induced velocity loss over one year.
    dV/year ~ integral of drag acceleration over one year.
    For circular orbits: dV_sk ~ (drag_accel) * seconds_per_year.
    But a simpler approach: dV = v * (rho * Cd * A / m) * v * dt / 2
    ... we just use the decay rate and orbital mechanics.

    Actually the simplest correct formula:
        dV_annual = (altitude_decay_m_per_year) * v / (2 * a)
    which reduces to dV = drag_accel * T_year.
    """
    if drag_params is None:
        drag_params = DragParameters()

    drag = compute_drag(altitude_km, drag_params, solar)

    # Delta-V per year = drag acceleration * seconds in a year
    seconds_per_year = 365.25 * 86400.0
    dv_per_year = drag.drag_acceleration_ms2 * seconds_per_year

    return dv_per_year


def station_keeping_budget(altitude_km: float,
                           mission_years: float,
                           propulsion: PropulsionSystem,
                           drag_params: Optional[DragParameters] = None,
                           solar: SolarActivity = SolarActivity.MODERATE,
                           maneuver_interval_days: float = 14.0,
                           ) -> StationKeepingBudget:
    """Compute full station-keeping budget for a mission.

    Args:
        altitude_km: Orbital altitude.
        mission_years: Mission lifetime.
        propulsion: Thruster system.
        drag_params: Satellite physical properties.
        solar: Solar activity level.
        maneuver_interval_days: Days between station-keeping burns.

    Returns:
        StationKeepingBudget with annual figures.
    """
    if drag_params is None:
        drag_params = DragParameters()

    dv_year = station_keeping_delta_v(altitude_km, solar, drag_params)
    prop_year = propulsion.propellant_mass_kg(dv_year, drag_params.mass_kg)
    burn_time_s = propulsion.burn_duration_s(dv_year, drag_params.mass_kg)
    maneuvers_year = int(math.ceil(365.25 / maneuver_interval_days))

    return StationKeepingBudget(
        altitude_km=altitude_km,
        delta_v_per_year_ms=dv_year,
        propellant_per_year_kg=prop_year,
        burn_time_per_year_hours=burn_time_s / 3600.0,
        maneuvers_per_year=maneuvers_year,
        propulsion_system=propulsion.name,
    )


# ---------------------------------------------------------------------------
# Deorbit planning
# ---------------------------------------------------------------------------

@dataclass
class DeorbitPlan:
    """End-of-life deorbit parameters."""
    current_altitude_km: float
    target_perigee_km: float
    delta_v_ms: float
    propellant_kg: float
    burn_duration_s: float
    natural_decay_years: float
    compliant_25yr: bool
    compliant_5yr: bool
    propulsion_system: str


def plan_deorbit(altitude_km: float,
                 propulsion: PropulsionSystem,
                 drag_params: Optional[DragParameters] = None,
                 solar: SolarActivity = SolarActivity.MODERATE,
                 target_perigee_km: float = 200.0,
                 ) -> DeorbitPlan:
    """Plan end-of-life deorbit maneuver.

    Computes the delta-V for a Hohmann-like transfer to lower the perigee
    to target_perigee_km (where atmospheric drag will do the rest).
    Also checks 25-year rule (ITU/IADC) and 5-year rule (FCC 2024).
    """
    if drag_params is None:
        drag_params = DragParameters()

    r1 = (EARTH_RADIUS_KM + altitude_km) * 1000.0  # m
    r2 = (EARTH_RADIUS_KM + target_perigee_km) * 1000.0  # m
    mu = EARTH_MU_KM3S2 * 1e9  # m^3/s^2

    # Circular velocity at current altitude
    v_circ = math.sqrt(mu / r1)

    # Transfer orbit semi-major axis
    a_transfer = (r1 + r2) / 2.0

    # Velocity at apoapsis of transfer orbit (which is our current altitude)
    v_transfer = math.sqrt(mu * (2.0 / r1 - 1.0 / a_transfer))

    # Delta-V: slow down from circular to transfer orbit
    dv = abs(v_circ - v_transfer)

    prop_mass = propulsion.propellant_mass_kg(dv, drag_params.mass_kg)
    burn_time = propulsion.burn_duration_s(dv, drag_params.mass_kg)

    # Natural decay time (without maneuver)
    drag = compute_drag(altitude_km, drag_params, solar)
    natural_decay_years = drag.days_to_deorbit / 365.25

    return DeorbitPlan(
        current_altitude_km=altitude_km,
        target_perigee_km=target_perigee_km,
        delta_v_ms=dv,
        propellant_kg=prop_mass,
        burn_duration_s=burn_time,
        natural_decay_years=natural_decay_years,
        compliant_25yr=natural_decay_years <= 25.0,
        compliant_5yr=natural_decay_years <= 5.0,
        propulsion_system=propulsion.name,
    )


# ---------------------------------------------------------------------------
# Compute operations impact
# ---------------------------------------------------------------------------

@dataclass
class ComputeImpact:
    """Impact of propulsion operations on compute availability."""
    maneuvers_per_year: int
    avg_downtime_per_maneuver_min: float
    annual_downtime_hours: float
    compute_availability_pct: float
    recommended_maneuver_timing: str
    total_mission_downtime_hours: float


def compute_operations_impact(
    altitude_km: float,
    mission_years: float,
    propulsion: PropulsionSystem,
    drag_params: Optional[DragParameters] = None,
    solar: SolarActivity = SolarActivity.MODERATE,
    maneuver_interval_days: float = 14.0,
    settle_time_min: float = 15.0,
) -> ComputeImpact:
    """Calculate how much compute time is lost to propulsion maneuvers.

    During a station-keeping burn:
    1. Attitude slew to thrust direction (~5 min)
    2. Thruster firing (variable, depends on delta-V per burn)
    3. Attitude settle / re-acquire pointing (~5-10 min)
    4. Power recovery if using electric propulsion (thrusters take power
       away from compute)

    Recommendation: schedule burns during eclipse (already power-limited)
    to minimize compute impact.
    """
    if drag_params is None:
        drag_params = DragParameters()

    budget = station_keeping_budget(
        altitude_km, mission_years, propulsion, drag_params, solar, maneuver_interval_days
    )

    # Delta-V per maneuver
    dv_per_maneuver = budget.delta_v_per_year_ms / budget.maneuvers_per_year
    burn_time_per_maneuver_s = propulsion.burn_duration_s(dv_per_maneuver, drag_params.mass_kg)

    # Total downtime per maneuver = attitude slew + burn + settle
    slew_time_min = 5.0
    burn_time_min = burn_time_per_maneuver_s / 60.0
    downtime_per_maneuver_min = slew_time_min + burn_time_min + settle_time_min

    annual_downtime_hours = (downtime_per_maneuver_min * budget.maneuvers_per_year) / 60.0
    hours_per_year = 365.25 * 24.0
    availability = (hours_per_year - annual_downtime_hours) / hours_per_year * 100.0

    # Recommend eclipse maneuvers for electric propulsion (saves compute power)
    if propulsion.prop_type == PropulsionType.HALL_THRUSTER:
        timing = "During eclipse — thruster power draw (%.0f W) offsets compute anyway" % propulsion.power_draw_w
    else:
        timing = "During eclipse — minimizes attitude disturbance impact on compute"

    total_downtime = annual_downtime_hours * mission_years

    # Add deorbit downtime (end-of-life)
    deorbit = plan_deorbit(altitude_km, propulsion, drag_params, solar)
    deorbit_downtime_hours = deorbit.burn_duration_s / 3600.0
    total_downtime += deorbit_downtime_hours

    return ComputeImpact(
        maneuvers_per_year=budget.maneuvers_per_year,
        avg_downtime_per_maneuver_min=round(downtime_per_maneuver_min, 1),
        annual_downtime_hours=round(annual_downtime_hours, 2),
        compute_availability_pct=round(availability, 4),
        recommended_maneuver_timing=timing,
        total_mission_downtime_hours=round(total_downtime, 2),
    )


# ---------------------------------------------------------------------------
# Constellation-level analysis
# ---------------------------------------------------------------------------

@dataclass
class ConstellationPropulsionSummary:
    """Propulsion budget for an entire constellation."""
    n_satellites: int
    altitude_km: float
    mission_years: float
    propulsion_system: str
    solar_activity: str

    # Per-satellite
    sk_delta_v_per_year_ms: float
    sk_propellant_per_year_kg: float
    deorbit_delta_v_ms: float
    deorbit_propellant_kg: float
    total_delta_v_ms: float
    total_propellant_kg: float

    # Constellation totals
    constellation_propellant_kg: float
    constellation_annual_downtime_hours: float
    per_sat_availability_pct: float

    # Compliance
    compliant_25yr: bool
    compliant_5yr: bool
    natural_decay_years: float


def analyze_constellation(
    n_satellites: int,
    altitude_km: float,
    mission_years: float,
    propulsion: Optional[PropulsionSystem] = None,
    drag_params: Optional[DragParameters] = None,
    solar: SolarActivity = SolarActivity.MODERATE,
) -> ConstellationPropulsionSummary:
    """Full propulsion analysis for a satellite constellation.

    Computes station-keeping budget, deorbit plan, and compute impact
    for the whole constellation.
    """
    if propulsion is None:
        propulsion = HALL_THRUSTER
    if drag_params is None:
        drag_params = DragParameters()

    # Station-keeping
    dv_year = station_keeping_delta_v(altitude_km, solar, drag_params)
    sk_prop_year = propulsion.propellant_mass_kg(dv_year, drag_params.mass_kg)

    # Deorbit
    deorbit = plan_deorbit(altitude_km, propulsion, drag_params, solar)

    # Total per satellite
    total_sk_dv = dv_year * mission_years
    total_sk_prop = sk_prop_year * mission_years
    total_dv = total_sk_dv + deorbit.delta_v_ms
    total_prop = total_sk_prop + deorbit.propellant_kg

    # Compute impact
    impact = compute_operations_impact(
        altitude_km, mission_years, propulsion, drag_params, solar
    )

    return ConstellationPropulsionSummary(
        n_satellites=n_satellites,
        altitude_km=altitude_km,
        mission_years=mission_years,
        propulsion_system=propulsion.name,
        solar_activity=solar.value,
        sk_delta_v_per_year_ms=round(dv_year, 4),
        sk_propellant_per_year_kg=round(sk_prop_year, 4),
        deorbit_delta_v_ms=round(deorbit.delta_v_ms, 4),
        deorbit_propellant_kg=round(deorbit.propellant_kg, 4),
        total_delta_v_ms=round(total_dv, 4),
        total_propellant_kg=round(total_prop, 4),
        constellation_propellant_kg=round(total_prop * n_satellites, 2),
        constellation_annual_downtime_hours=round(
            impact.annual_downtime_hours * n_satellites, 2
        ),
        per_sat_availability_pct=impact.compute_availability_pct,
        compliant_25yr=deorbit.compliant_25yr,
        compliant_5yr=deorbit.compliant_5yr,
        natural_decay_years=round(deorbit.natural_decay_years, 1),
    )


# ---------------------------------------------------------------------------
# __main__ demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 72)
    print("PROPULSION & ORBIT MAINTENANCE — Constellation Analysis")
    print("=" * 72)

    # Configuration: 12-satellite constellation at 550 km, 5-year life
    N_SATS = 12
    ALT_KM = 550.0
    MISSION_YEARS = 5.0

    sat_params = DragParameters(mass_kg=260.0, cross_section_m2=4.0, drag_coefficient=2.2)

    # --- Atmospheric drag at 550 km ---
    print("\n--- Atmospheric Drag at %.0f km ---" % ALT_KM)
    for solar in SolarActivity:
        atmo = atmospheric_density(ALT_KM, solar)
        drag = compute_drag(ALT_KM, sat_params, solar)
        print(f"  Solar {solar.value:>8s}: rho={atmo.density_kg_m3:.2e} kg/m3, "
              f"drag={drag.drag_force_n:.4e} N, "
              f"decay={drag.altitude_decay_km_per_day:.4f} km/day, "
              f"deorbit in {drag.days_to_deorbit:.0f} days ({drag.days_to_deorbit/365.25:.1f} yr)")

    # --- Compare propulsion systems ---
    print("\n--- Propulsion System Comparison (550 km, moderate solar, 5-year mission) ---")
    print(f"  {'System':<28s} {'Isp(s)':>7s} {'Thrust':>10s} {'dV/yr(m/s)':>11s} "
          f"{'Prop/yr(kg)':>12s} {'Total(kg)':>10s} {'Deorbit(kg)':>11s}")
    print("  " + "-" * 95)

    for thruster in [HALL_THRUSTER, HYDRAZINE, COLD_GAS]:
        budget = station_keeping_budget(ALT_KM, MISSION_YEARS, thruster, sat_params)
        deorbit = plan_deorbit(ALT_KM, thruster, sat_params)
        total_prop = budget.propellant_per_year_kg * MISSION_YEARS + deorbit.propellant_kg

        thrust_str = "%.1f mN" % (thruster.thrust_n * 1000) if thruster.thrust_n < 1.0 else "%.1f N" % thruster.thrust_n
        print(f"  {thruster.name:<28s} {thruster.specific_impulse_s:>7.0f} {thrust_str:>10s} "
              f"{budget.delta_v_per_year_ms:>11.4f} {budget.propellant_per_year_kg:>12.4f} "
              f"{total_prop:>10.3f} {deorbit.propellant_kg:>11.4f}")

    # --- Full constellation analysis (Hall thruster) ---
    print("\n--- Constellation Analysis: %d satellites, %.0f km, %.0f-year life ---"
          % (N_SATS, ALT_KM, MISSION_YEARS))

    for solar in SolarActivity:
        summary = analyze_constellation(
            N_SATS, ALT_KM, MISSION_YEARS, HALL_THRUSTER, sat_params, solar
        )
        print(f"\n  Solar activity: {solar.value}")
        print(f"    Station-keeping dV:     {summary.sk_delta_v_per_year_ms:.4f} m/s/year")
        print(f"    Station-keeping prop:   {summary.sk_propellant_per_year_kg:.4f} kg/year/sat")
        print(f"    Deorbit dV:             {summary.deorbit_delta_v_ms:.2f} m/s")
        print(f"    Deorbit propellant:     {summary.deorbit_propellant_kg:.4f} kg/sat")
        print(f"    Total prop per sat:     {summary.total_propellant_kg:.3f} kg")
        print(f"    Constellation total:    {summary.constellation_propellant_kg:.1f} kg")
        print(f"    Compute availability:   {summary.per_sat_availability_pct:.4f}%")
        print(f"    Natural decay:          {summary.natural_decay_years} years")
        print(f"    25-year compliant:      {'YES' if summary.compliant_25yr else 'NO'}")
        print(f"    5-year compliant (FCC): {'YES' if summary.compliant_5yr else 'NO'}")

    # --- Compute impact detail ---
    print("\n--- Compute Operations Impact (Hall thruster, moderate solar) ---")
    impact = compute_operations_impact(ALT_KM, MISSION_YEARS, HALL_THRUSTER, sat_params)
    print(f"    Maneuvers per year:        {impact.maneuvers_per_year}")
    print(f"    Downtime per maneuver:     {impact.avg_downtime_per_maneuver_min} min")
    print(f"    Annual downtime:           {impact.annual_downtime_hours} hours/sat")
    print(f"    Total mission downtime:    {impact.total_mission_downtime_hours} hours/sat")
    print(f"    Compute availability:      {impact.compute_availability_pct}%")
    print(f"    Recommended timing:        {impact.recommended_maneuver_timing}")

    # --- Altitude comparison ---
    print("\n--- Altitude Comparison (Hall thruster, moderate solar, 5-year) ---")
    print(f"  {'Alt(km)':>8s} {'dV/yr(m/s)':>11s} {'Prop/yr(kg)':>12s} {'Decay(yr)':>10s} "
          f"{'25yr':>5s} {'5yr':>5s} {'Avail%':>8s}")
    print("  " + "-" * 65)

    for alt in [400, 500, 550, 600, 800, 1000, 1200]:
        s = analyze_constellation(1, float(alt), MISSION_YEARS, HALL_THRUSTER, sat_params)
        print(f"  {alt:>8d} {s.sk_delta_v_per_year_ms:>11.4f} {s.sk_propellant_per_year_kg:>12.4f} "
              f"{s.natural_decay_years:>10.1f} "
              f"{'Y' if s.compliant_25yr else 'N':>5s} "
              f"{'Y' if s.compliant_5yr else 'N':>5s} "
              f"{s.per_sat_availability_pct:>8.4f}")

    print("\n" + "=" * 72)
    print("Done.")
