"""Orbital debris environment modeling and collision avoidance.

Models the LEO debris environment, performs conjunction assessment,
plans collision avoidance maneuvers (CAMs), and assesses constellation-level
impact including Kessler syndrome risk.

Based on simplified NASA ORDEM (Orbital Debris Engineering Model) data
for spatial density by altitude band.

References:
    - NASA ORDEM 3.1 (2020) for debris flux/density profiles
    - ESA MASTER model for cross-validation of density peaks
    - IADC Space Debris Mitigation Guidelines (2020)
    - Kessler & Cour-Palais (1978) for collision cascade model
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EARTH_RADIUS_KM = 6371.0
MU_EARTH = 3.986004418e5  # km^3/s^2  (gravitational parameter)
SECONDS_PER_YEAR = 365.25 * 86400


# ---------------------------------------------------------------------------
# 1. Debris Environment Model  (simplified ORDEM)
# ---------------------------------------------------------------------------

# Spatial density of tracked objects (>10 cm) by altitude band.
# Units: objects / km^3
# Source: Simplified fit to NASA ORDEM 3.1 output for 2025 epoch.
# The two prominent peaks at ~800 km and ~1400 km are well-documented
# in the literature (post-Cosmos 2251 / Fengyun-1C altitude bands).
_ORDEM_DENSITY_TABLE: List[Tuple[float, float, float]] = [
    # (alt_low_km, alt_high_km, density_per_km3 for >10cm objects)
    (200, 400, 1.0e-9),
    (400, 500, 3.0e-9),
    (500, 600, 5.0e-9),
    (600, 700, 8.0e-9),
    (700, 800, 2.0e-8),
    (800, 900, 4.0e-8),   # peak 1 — Cosmos/Fengyun debris
    (900, 1000, 2.5e-8),
    (1000, 1100, 1.5e-8),
    (1100, 1200, 1.2e-8),
    (1200, 1300, 1.5e-8),
    (1300, 1400, 2.5e-8),
    (1400, 1500, 3.5e-8),  # peak 2
    (1500, 1600, 2.0e-8),
    (1600, 1800, 1.0e-8),
    (1800, 2000, 5.0e-9),
]

# Untracked (<10 cm) multiplier — for every tracked object there are
# roughly 30-100x lethal-but-untracked fragments (1-10 cm).
_UNTRACKED_MULTIPLIER = 50.0

# Total tracked catalogue size (as of ~2025).
TRACKED_OBJECTS_TOTAL = 30_000

# Average cross-section of a typical 3U-12U compute satellite [km^2].
TYPICAL_SAT_CROSS_SECTION_KM2 = 1.0e-6  # ~1 m^2 in km^2

# Average relative collision speed in LEO [km/s].
# Head-on: ~14 km/s, co-planar: ~1 km/s.  Weighted average ~10 km/s.
AVG_COLLISION_SPEED_KMS = 10.0


@dataclass
class DebrisEnvironment:
    """Debris spatial density model for a given altitude."""

    altitude_km: float
    tracked_density_per_km3: float   # >10 cm objects
    total_density_per_km3: float     # including untracked lethal (>1 cm)

    @staticmethod
    def at_altitude(alt_km: float) -> "DebrisEnvironment":
        """Look up (interpolate) debris density at a given altitude."""
        tracked = _interp_density(alt_km)
        total = tracked + tracked * _UNTRACKED_MULTIPLIER
        return DebrisEnvironment(
            altitude_km=alt_km,
            tracked_density_per_km3=tracked,
            total_density_per_km3=total,
        )

    def collision_probability_per_year(
        self,
        cross_section_km2: float = TYPICAL_SAT_CROSS_SECTION_KM2,
        rel_speed_kms: float = AVG_COLLISION_SPEED_KMS,
    ) -> float:
        """Annual collision probability for a single satellite (kinetic theory).

        P = density * cross_section * relative_speed * seconds_per_year
        This is the standard "flux times area" approach from NASA.
        """
        flux = self.total_density_per_km3 * rel_speed_kms  # objects / km^2 / s
        return flux * cross_section_km2 * SECONDS_PER_YEAR


def _interp_density(alt_km: float) -> float:
    """Linearly interpolate tracked debris density from the table."""
    if alt_km < _ORDEM_DENSITY_TABLE[0][0]:
        return _ORDEM_DENSITY_TABLE[0][2]
    if alt_km > _ORDEM_DENSITY_TABLE[-1][1]:
        return _ORDEM_DENSITY_TABLE[-1][2]

    for low, high, density in _ORDEM_DENSITY_TABLE:
        if low <= alt_km < high:
            return density
    return _ORDEM_DENSITY_TABLE[-1][2]


# ---------------------------------------------------------------------------
# 2. Conjunction Assessment
# ---------------------------------------------------------------------------

@dataclass
class ConjunctionEvent:
    """Result of a conjunction (close approach) screening."""

    object_a_id: str
    object_b_id: str
    tca_seconds: float           # time of closest approach from epoch [s]
    miss_distance_km: float
    relative_speed_kms: float
    severity: str                # "nominal", "warning", "critical"

    @property
    def miss_distance_m(self) -> float:
        return self.miss_distance_km * 1000.0


@dataclass
class OrbitalState:
    """Simplified Keplerian + Cartesian state for conjunction math."""

    object_id: str
    pos_km: np.ndarray       # [x, y, z] ECI
    vel_kms: np.ndarray      # [vx, vy, vz] ECI
    altitude_km: float = 0.0


def closest_approach(
    state_a: OrbitalState,
    state_b: OrbitalState,
    dt_seconds: float = 1.0,
    window_seconds: float = 600.0,
    warning_km: float = 1.0,
    critical_km: float = 0.2,
) -> ConjunctionEvent:
    """Brute-force closest approach over a short window (linearised).

    For a real system you would use smart filtering (Alfano's method,
    covariance-based probability of collision, etc.).  This simplified
    version propagates linearly and finds the minimum distance.
    """
    best_dist = float("inf")
    best_t = 0.0

    t = 0.0
    while t <= window_seconds:
        pa = state_a.pos_km + state_a.vel_kms * t
        pb = state_b.pos_km + state_b.vel_kms * t
        d = float(np.linalg.norm(pa - pb))
        if d < best_dist:
            best_dist = d
            best_t = t
        t += dt_seconds

    rel_v = float(np.linalg.norm(state_a.vel_kms - state_b.vel_kms))
    if best_dist < critical_km:
        severity = "critical"
    elif best_dist < warning_km:
        severity = "warning"
    else:
        severity = "nominal"

    return ConjunctionEvent(
        object_a_id=state_a.object_id,
        object_b_id=state_b.object_id,
        tca_seconds=best_t,
        miss_distance_km=best_dist,
        relative_speed_kms=rel_v,
        severity=severity,
    )


def screen_constellation(
    constellation: List[OrbitalState],
    debris_catalog: List[OrbitalState],
    warning_km: float = 1.0,
    critical_km: float = 0.2,
) -> List[ConjunctionEvent]:
    """Screen every constellation satellite against the debris catalog.

    Returns only warning or critical conjunctions.
    """
    events: List[ConjunctionEvent] = []
    for sat in constellation:
        for obj in debris_catalog:
            evt = closest_approach(
                sat, obj,
                warning_km=warning_km,
                critical_km=critical_km,
            )
            if evt.severity != "nominal":
                events.append(evt)
    return events


# ---------------------------------------------------------------------------
# 3. Collision Avoidance Maneuver (CAM) Planner
# ---------------------------------------------------------------------------

@dataclass
class CAMPlan:
    """Collision avoidance maneuver plan."""

    conjunction: ConjunctionEvent
    delta_v_ms: float              # required delta-V [m/s]
    propellant_kg: float           # propellant consumed [kg]
    maneuver_duration_s: float     # burn + settling time [s]
    compute_downtime_s: float      # total ops downtime [s]
    direction: str                 # "in-track" or "radial"


def plan_cam(
    conjunction: ConjunctionEvent,
    satellite_mass_kg: float = 150.0,
    isp_s: float = 220.0,
    lead_time_hours: float = 6.0,
    settling_time_s: float = 600.0,
) -> CAMPlan:
    """Plan a collision avoidance maneuver.

    Uses the along-track displacement approach: a small delta-V applied
    hours before TCA shifts the satellite's along-track position by
    enough to increase miss distance.

    delta_x ~ delta_v * lead_time  (simplified — real dynamics use
    Clohessy-Wiltshire relative motion equations).

    We target moving the satellite by 1 km beyond the miss distance
    to provide margin.
    """
    target_shift_km = max(1.0, 1.0 - conjunction.miss_distance_km + 0.5)
    lead_time_s = lead_time_hours * 3600.0

    # delta_v = displacement / lead_time  (linearised)
    delta_v_kms = target_shift_km / lead_time_s
    delta_v_ms = delta_v_kms * 1000.0  # convert to m/s

    # Tsiolkovsky: delta_m = m * (1 - exp(-dv / (Isp * g0)))
    g0 = 9.80665  # m/s^2
    mass_fraction = 1.0 - math.exp(-delta_v_ms / (isp_s * g0))
    propellant_kg = satellite_mass_kg * mass_fraction

    # Burn duration estimate (assume 0.5 N thruster for small sat)
    thrust_n = 0.5
    burn_time_s = (propellant_kg * g0 * isp_s) / thrust_n if thrust_n > 0 else 0.0

    maneuver_duration_s = burn_time_s + settling_time_s
    # Compute downtime: slewing + burn + settling + return to ops attitude
    compute_downtime_s = maneuver_duration_s + 300.0  # 5 min for attitude recovery

    return CAMPlan(
        conjunction=conjunction,
        delta_v_ms=delta_v_ms,
        propellant_kg=propellant_kg,
        maneuver_duration_s=maneuver_duration_s,
        compute_downtime_s=compute_downtime_s,
        direction="in-track",
    )


def typical_cams_per_year(
    altitude_km: float,
    cross_section_km2: float = TYPICAL_SAT_CROSS_SECTION_KM2,
) -> float:
    """Estimate how many CAMs a satellite performs per year.

    Based on empirical data: ISS (~400 km) does ~2-3/year, Starlink
    (~550 km) averages ~0.5-1/sat/year but with 5000+ sats the fleet
    total is huge.  Higher altitudes with more debris see more.

    We use: CAMs/year ~ k * tracked_density * cross_section * orbital_speed
    Calibrated so 550 km gives ~1-2 CAMs/year and 800 km gives ~4-6.
    """
    env = DebrisEnvironment.at_altitude(altitude_km)
    # Use tracked density (only tracked objects trigger CAMs)
    orbital_speed = math.sqrt(MU_EARTH / (EARTH_RADIUS_KM + altitude_km))  # km/s
    flux = env.tracked_density_per_km3 * orbital_speed  # per km^2 per s

    # The "action radius" for conjunction screening is much larger than
    # the physical cross-section — typically a box of ~5-10 km.
    screening_volume_km2 = 25.0  # ~5x5 km screening area
    raw_encounters_per_year = flux * screening_volume_km2 * SECONDS_PER_YEAR

    # Only a fraction of encounters require a maneuver (probability
    # of collision > threshold after covariance analysis).
    # Empirical: roughly 0.5-2% of flagged conjunctions need a CAM.
    cam_fraction = 0.01
    return raw_encounters_per_year * cam_fraction


# ---------------------------------------------------------------------------
# 4. Constellation Impact Assessment
# ---------------------------------------------------------------------------

@dataclass
class ConstellationDebrisAssessment:
    """Full debris impact assessment for a constellation."""

    altitude_km: float
    n_satellites: int
    debris_env: DebrisEnvironment

    # Per-satellite metrics
    collision_prob_per_sat_per_year: float
    cams_per_sat_per_year: float
    propellant_per_cam_kg: float
    downtime_per_cam_hours: float

    # Fleet-wide metrics
    fleet_collision_prob_per_year: float
    fleet_cams_per_year: float
    fleet_propellant_per_year_kg: float
    fleet_downtime_per_year_hours: float

    # Mission impact
    mission_lifetime_propellant_kg: float  # total CAM propellant over mission
    compute_availability_loss_pct: float   # % of compute time lost to CAMs

    # Kessler risk
    kessler_critical_population: int       # constellation size where debris grows
    kessler_risk_level: str                # "low", "moderate", "high"


def assess_constellation(
    altitude_km: float,
    n_satellites: int,
    mission_years: float = 5.0,
    satellite_mass_kg: float = 150.0,
    isp_s: float = 220.0,
) -> ConstellationDebrisAssessment:
    """Full debris risk assessment for an orbital compute constellation."""
    env = DebrisEnvironment.at_altitude(altitude_km)

    # Per-satellite collision probability
    col_prob = env.collision_probability_per_year()

    # CAMs per satellite per year
    cams_per_sat = typical_cams_per_year(altitude_km)

    # Typical CAM cost (for a "warning" level conjunction)
    sample_conjunction = ConjunctionEvent(
        object_a_id="SAT", object_b_id="DEB",
        tca_seconds=0, miss_distance_km=0.15,
        relative_speed_kms=10.0, severity="warning",
    )
    sample_cam = plan_cam(
        sample_conjunction,
        satellite_mass_kg=satellite_mass_kg,
        isp_s=isp_s,
    )

    # Fleet-wide
    fleet_col_prob = 1.0 - (1.0 - col_prob) ** n_satellites
    fleet_cams = cams_per_sat * n_satellites
    fleet_propellant = fleet_cams * sample_cam.propellant_kg
    fleet_downtime_hours = fleet_cams * sample_cam.compute_downtime_s / 3600.0

    # Mission lifetime propellant
    mission_propellant = fleet_propellant * mission_years

    # Compute availability loss
    total_sat_hours_per_year = n_satellites * 8766.0  # hours/year
    avail_loss = (fleet_downtime_hours / total_sat_hours_per_year) * 100.0

    # Kessler syndrome assessment
    kessler_pop = kessler_critical_size(altitude_km)
    if n_satellites > kessler_pop * 0.8:
        kessler_risk = "high"
    elif n_satellites > kessler_pop * 0.3:
        kessler_risk = "moderate"
    else:
        kessler_risk = "low"

    return ConstellationDebrisAssessment(
        altitude_km=altitude_km,
        n_satellites=n_satellites,
        debris_env=env,
        collision_prob_per_sat_per_year=col_prob,
        cams_per_sat_per_year=cams_per_sat,
        propellant_per_cam_kg=sample_cam.propellant_kg,
        downtime_per_cam_hours=sample_cam.compute_downtime_s / 3600.0,
        fleet_collision_prob_per_year=fleet_col_prob,
        fleet_cams_per_year=fleet_cams,
        fleet_propellant_per_year_kg=fleet_propellant,
        fleet_downtime_per_year_hours=fleet_downtime_hours,
        mission_lifetime_propellant_kg=mission_propellant,
        compute_availability_loss_pct=avail_loss,
        kessler_critical_population=kessler_pop,
        kessler_risk_level=kessler_risk,
    )


# ---------------------------------------------------------------------------
# 5. Kessler Syndrome Risk Model
# ---------------------------------------------------------------------------

def atmospheric_drag_removal_rate(altitude_km: float) -> float:
    """Approximate debris removal rate due to atmospheric drag [fraction/year].

    Below ~600 km, drag is effective — debris decays in years.
    Above ~800 km, debris persists for centuries.
    """
    if altitude_km < 300:
        return 1.0       # decays within ~1 year
    elif altitude_km < 400:
        return 0.3        # ~3 year lifetime
    elif altitude_km < 500:
        return 0.1        # ~10 year lifetime
    elif altitude_km < 600:
        return 0.03       # ~30 year lifetime
    elif altitude_km < 700:
        return 0.01       # ~100 years
    elif altitude_km < 800:
        return 0.003      # ~300 years
    elif altitude_km < 1000:
        return 0.001      # ~1000 years
    else:
        return 0.0002     # effectively permanent (>5000 years)


def collision_fragment_rate(
    altitude_km: float,
    n_objects: int,
    cross_section_km2: float = TYPICAL_SAT_CROSS_SECTION_KM2,
) -> float:
    """Estimate new debris fragments generated per year from collisions.

    Each catastrophic collision produces ~1000-3000 trackable fragments.
    Rate = n_objects * density * cross_section * velocity * fragments_per_collision
    """
    env = DebrisEnvironment.at_altitude(altitude_km)
    orbital_speed = math.sqrt(MU_EARTH / (EARTH_RADIUS_KM + altitude_km))
    collision_rate = (
        n_objects
        * env.total_density_per_km3
        * cross_section_km2
        * orbital_speed
        * SECONDS_PER_YEAR
    )
    fragments_per_collision = 1500  # average for catastrophic collision
    return collision_rate * fragments_per_collision


def kessler_critical_size(
    altitude_km: float,
    cross_section_km2: float = TYPICAL_SAT_CROSS_SECTION_KM2,
) -> int:
    """Find the constellation size where debris generation exceeds removal.

    This is the tipping point for Kessler syndrome at this altitude.
    We model the balance between:
      - Fragment generation: constellation-on-debris collisions producing
        ~1500 trackable fragments each
      - Fragment removal: atmospheric drag clearing debris at altitude-
        dependent rates

    The critical size N satisfies:
      N * P_collision_tracked * fragments_per_collision = removal_rate * existing_pop

    Note: this is a simplified steady-state model. Real Kessler dynamics
    involve feedback loops (new debris raises future collision rates).
    """
    env = DebrisEnvironment.at_altitude(altitude_km)
    removal_rate = atmospheric_drag_removal_rate(altitude_km)

    # Current tracked debris population in this altitude band (approximate)
    # Use a 100 km thick shell at this altitude
    shell_volume = 4.0 * math.pi * (EARTH_RADIUS_KM + altitude_km) ** 2 * 100.0
    current_tracked = env.tracked_density_per_km3 * shell_volume

    # Objects removed per year by drag
    objects_removed_per_year = current_tracked * removal_rate

    if objects_removed_per_year < 1e-12:
        # Negligible drag at this altitude — even small constellations
        # contribute to long-term debris growth.
        return max(50, int(100 * (1.0 - min(altitude_km, 2000) / 2000)))

    # Collision rate per constellation satellite vs tracked debris
    # Use tracked density only (large objects cause catastrophic collisions)
    orbital_speed = math.sqrt(MU_EARTH / (EARTH_RADIUS_KM + altitude_km))

    # Effective cross-section for catastrophic collision is larger than
    # geometric — includes the debris object size (~1-5 m^2).
    effective_cross_section = cross_section_km2 * 5.0  # ~5 m^2 combined

    collision_rate_per_sat = (
        env.tracked_density_per_km3
        * effective_cross_section
        * orbital_speed
        * SECONDS_PER_YEAR
    )

    fragments_per_collision = 1500

    # N_crit: N * collision_rate * fragments = removal_rate * population
    fragments_per_sat_per_year = collision_rate_per_sat * fragments_per_collision

    if fragments_per_sat_per_year < 1e-20:
        return 1_000_000  # effectively no Kessler risk

    n_crit = int(objects_removed_per_year / fragments_per_sat_per_year)
    return max(n_crit, 10)


# ---------------------------------------------------------------------------
# __main__ — Risk assessment demo
# ---------------------------------------------------------------------------

def _format_assessment(a: ConstellationDebrisAssessment) -> str:
    lines = [
        f"{'=' * 65}",
        f"  DEBRIS RISK ASSESSMENT — {a.n_satellites} satellites @ {a.altitude_km:.0f} km",
        f"{'=' * 65}",
        "",
        "  DEBRIS ENVIRONMENT",
        f"    Tracked density (>10cm):  {a.debris_env.tracked_density_per_km3:.2e} obj/km³",
        f"    Total lethal density:     {a.debris_env.total_density_per_km3:.2e} obj/km³",
        "",
        "  PER-SATELLITE (annual)",
        f"    Collision probability:    {a.collision_prob_per_sat_per_year:.2e}",
        f"    Expected CAMs/year:       {a.cams_per_sat_per_year:.1f}",
        f"    Propellant per CAM:       {a.propellant_per_cam_kg * 1000:.1f} g",
        f"    Downtime per CAM:         {a.downtime_per_cam_hours:.2f} hrs",
        "",
        "  FLEET-WIDE (annual)",
        f"    Fleet collision prob:     {a.fleet_collision_prob_per_year:.4f}  ({a.fleet_collision_prob_per_year * 100:.2f}%)",
        f"    Total CAMs/year:          {a.fleet_cams_per_year:.0f}",
        f"    Propellant budget/year:   {a.fleet_propellant_per_year_kg:.2f} kg",
        f"    Compute downtime/year:    {a.fleet_downtime_per_year_hours:.1f} hrs",
        f"    Availability loss:        {a.compute_availability_loss_pct:.4f}%",
        "",
        "  KESSLER SYNDROME",
        f"    Critical population:      {a.kessler_critical_population:,} objects at this altitude",
        f"    Drag removal rate:        {atmospheric_drag_removal_rate(a.altitude_km):.4f} /year",
        f"    Risk level:               {a.kessler_risk_level.upper()}",
        f"{'=' * 65}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print()
    print("ORBITAL DEBRIS AVOIDANCE MODULE — Constellation Risk Assessment")
    print("=" * 65)

    # --- Scenario 1: 100 sats at 550 km (Starlink-like altitude) ---
    a550 = assess_constellation(altitude_km=550, n_satellites=100)
    print()
    print(_format_assessment(a550))

    # --- Scenario 2: 100 sats at 1200 km (higher LEO, more debris) ---
    a1200 = assess_constellation(altitude_km=1200, n_satellites=100)
    print()
    print(_format_assessment(a1200))

    # --- Comparison ---
    print()
    print("COMPARISON: 550 km vs 1200 km")
    print("-" * 50)
    ratio_col = a1200.collision_prob_per_sat_per_year / max(a550.collision_prob_per_sat_per_year, 1e-30)
    ratio_cam = a1200.cams_per_sat_per_year / max(a550.cams_per_sat_per_year, 1e-30)
    print(f"  Collision risk ratio (1200/550): {ratio_col:.1f}x")
    print(f"  CAM frequency ratio (1200/550): {ratio_cam:.1f}x")
    print(f"  Kessler risk: 550km={a550.kessler_risk_level}, 1200km={a1200.kessler_risk_level}")
    drag_550 = atmospheric_drag_removal_rate(550)
    drag_1200 = atmospheric_drag_removal_rate(1200)
    print(f"  Natural debris removal: 550km={drag_550:.3f}/yr, 1200km={drag_1200:.4f}/yr")
    print()
    if drag_1200 < 0.01:
        print("  ⚠ WARNING: At 1200 km, debris persists for millennia.")
        print("    A constellation failure or collision creates PERMANENT debris.")
        print("    550 km is strongly preferred for responsible operations.")
    print()

    # --- Conjunction demo ---
    print("CONJUNCTION ASSESSMENT DEMO")
    print("-" * 50)
    sat = OrbitalState(
        object_id="COMPUTE-SAT-1",
        pos_km=np.array([6921.0, 0.0, 0.0]),
        vel_kms=np.array([0.0, 7.6, 0.0]),
        altitude_km=550.0,
    )
    debris_obj = OrbitalState(
        object_id="DEBRIS-44231",
        pos_km=np.array([6921.05, 0.1, 0.0]),
        vel_kms=np.array([0.0, -7.6, 0.0]),  # head-on
        altitude_km=550.0,
    )
    event = closest_approach(sat, debris_obj)
    print(f"  Objects: {event.object_a_id} vs {event.object_b_id}")
    print(f"  Miss distance: {event.miss_distance_m:.0f} m")
    print(f"  Relative speed: {event.relative_speed_kms:.1f} km/s")
    print(f"  Severity: {event.severity.upper()}")

    if event.severity != "nominal":
        cam = plan_cam(event)
        print(f"\n  CAM PLAN:")
        print(f"    Delta-V required: {cam.delta_v_ms:.3f} m/s")
        print(f"    Propellant cost:  {cam.propellant_kg * 1000:.1f} g")
        print(f"    Compute downtime: {cam.compute_downtime_s / 60:.1f} min")
        print(f"    Maneuver type:    {cam.direction}")

    print()
