"""Comprehensive test suite for orbital-compute core modules.

Tests orbit, power, thermal, scheduler, ground_stations, and workloads
using only the stdlib unittest framework (no pytest needed).
"""

import math
import unittest
from datetime import datetime, timedelta, timezone

import numpy as np

from orbital_compute.orbit import (
    Satellite, SatPosition, sun_position_eci, is_in_eclipse,
    starlink_shell_1_sample, EARTH_RADIUS_KM, predict_eclipse_windows,
)
from orbital_compute.power import PowerModel, PowerConfig, PowerState
from orbital_compute.thermal import ThermalModel, ThermalConfig, ThermalState, STEFAN_BOLTZMANN
from orbital_compute.scheduler import (
    OrbitalScheduler, ComputeJob, JobType, JobStatus, ScheduleDecision,
)
from orbital_compute.ground_stations import (
    GroundStation, find_contact_windows, elevation_angle,
    DEFAULT_GROUND_STATIONS, _lla_to_ecef, _ecef_to_eci,
)
from orbital_compute.workloads import (
    WORKLOAD_CATALOG, WorkloadGenerator, create_job,
    image_classification, change_detection, object_tracking,
    llm_inference, image_generation, weather_model, climate_analysis,
    sar_processing, signal_analysis,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
T0 = datetime(2026, 3, 26, 0, 0, 0, tzinfo=timezone.utc)


def _make_sat(idx: int = 0) -> Satellite:
    """Return the first synthetic Starlink satellite."""
    return starlink_shell_1_sample(max(idx + 1, 1))[idx]


def _simple_job(job_id="J1", power=300.0, duration=600.0, priority=5,
                checkpointable=True, min_battery_pct=0.30) -> ComputeJob:
    return ComputeJob(
        job_id=job_id,
        name=f"test-{job_id}",
        power_watts=power,
        duration_seconds=duration,
        priority=priority,
        checkpointable=checkpointable,
        min_battery_pct=min_battery_pct,
    )


# ===================================================================
# orbit.py tests
# ===================================================================
class TestOrbit(unittest.TestCase):
    """Tests for orbital mechanics module."""

    def test_position_returns_valid_coordinates(self):
        """Satellite position at known time returns valid coordinates."""
        sat = _make_sat()
        pos = sat.position_at(T0)

        self.assertIsInstance(pos, SatPosition)
        # Position vector should be roughly LEO distance from Earth center
        r = math.sqrt(pos.x_km**2 + pos.y_km**2 + pos.z_km**2)
        self.assertGreater(r, EARTH_RADIUS_KM, "Satellite must be above Earth surface")
        self.assertLess(r, EARTH_RADIUS_KM + 1000, "LEO satellite should be <1000 km alt")
        # Altitude cross-check
        self.assertAlmostEqual(pos.altitude_km, r - EARTH_RADIUS_KM, delta=50)

    def test_position_lat_lon_bounds(self):
        """Latitude in [-90,90], longitude in [-180,180]."""
        sat = _make_sat()
        for minutes in range(0, 120, 10):
            t = T0 + timedelta(minutes=minutes)
            pos = sat.position_at(t)
            self.assertGreaterEqual(pos.lat_deg, -90)
            self.assertLessEqual(pos.lat_deg, 90)
            self.assertGreaterEqual(pos.lon_deg, -180)
            self.assertLessEqual(pos.lon_deg, 180)

    def test_eclipse_behind_earth(self):
        """Satellite behind Earth (anti-sun side) is in eclipse."""
        sun_pos = np.array([1.0e8, 0.0, 0.0])  # Sun along +X
        # Satellite on opposite side at LEO altitude
        sat_pos = np.array([-(EARTH_RADIUS_KM + 550), 0.0, 0.0])
        self.assertTrue(is_in_eclipse(sat_pos, sun_pos))

    def test_no_eclipse_in_front_of_earth(self):
        """Satellite between Earth and Sun is NOT in eclipse."""
        sun_pos = np.array([1.0e8, 0.0, 0.0])  # Sun along +X
        sat_pos = np.array([EARTH_RADIUS_KM + 550, 0.0, 0.0])  # Same side as sun
        self.assertFalse(is_in_eclipse(sat_pos, sun_pos))

    def test_no_eclipse_far_from_shadow(self):
        """Satellite well off the Earth-Sun line is not in eclipse."""
        sun_pos = np.array([1.0e8, 0.0, 0.0])
        # Behind Earth in X, but far off in Y (outside shadow cylinder)
        sat_pos = np.array([-(EARTH_RADIUS_KM + 550), EARTH_RADIUS_KM * 3, 0.0])
        self.assertFalse(is_in_eclipse(sat_pos, sun_pos))

    def test_sun_position_reasonable(self):
        """Sun position magnitude is approximately 1 AU."""
        sun = sun_position_eci(T0)
        dist_km = np.linalg.norm(sun)
        au_km = 149597870.7
        self.assertAlmostEqual(dist_km / au_km, 1.0, delta=0.02,
                               msg="Sun distance should be ~1 AU")

    def test_sun_position_changes_over_year(self):
        """Sun direction changes significantly over 6 months."""
        sun_mar = sun_position_eci(T0)
        sun_sep = sun_position_eci(T0 + timedelta(days=182))
        cos_angle = np.dot(sun_mar, sun_sep) / (np.linalg.norm(sun_mar) * np.linalg.norm(sun_sep))
        # Should be roughly opposite (~-1)
        self.assertLess(cos_angle, 0.0, "Sun should be on opposite side after 6 months")

    def test_synthetic_tle_generation_produces_valid_satellites(self):
        """starlink_shell_1_sample creates working Satellite objects."""
        sats = starlink_shell_1_sample(12)
        self.assertEqual(len(sats), 12)
        for sat in sats:
            self.assertIsInstance(sat, Satellite)
            pos = sat.position_at(T0)
            self.assertGreater(pos.altitude_km, 400)
            self.assertLess(pos.altitude_km, 700)

    def test_predict_eclipse_windows(self):
        """Eclipse prediction returns non-overlapping windows."""
        sat = _make_sat()
        windows = predict_eclipse_windows(sat, T0, duration_hours=6.0, step_seconds=60)
        # LEO satellite should have eclipse windows
        # (could be 0 in edge cases, but very unlikely over 6 hours)
        for start, end in windows:
            self.assertLess(start, end)
        # Windows should not overlap
        for i in range(len(windows) - 1):
            self.assertLessEqual(windows[i][1], windows[i + 1][0])


# ===================================================================
# power.py tests
# ===================================================================
class TestPower(unittest.TestCase):
    """Tests for power subsystem model."""

    def test_battery_charges_in_sunlight(self):
        """Battery SoC increases when in sunlight with no compute load."""
        cfg = PowerConfig(battery_initial_pct=0.50)
        pm = PowerModel(cfg)
        initial = pm.battery_wh

        state = pm.step(3600.0, in_eclipse=False, compute_load_w=0.0)

        self.assertGreater(pm.battery_wh, initial)
        self.assertGreater(state.solar_output_w, 0)

    def test_battery_drains_in_eclipse(self):
        """Battery SoC decreases when in eclipse (housekeeping load)."""
        cfg = PowerConfig(battery_initial_pct=0.80)
        pm = PowerModel(cfg)
        initial = pm.battery_wh

        pm.step(3600.0, in_eclipse=True, compute_load_w=0.0)

        self.assertLess(pm.battery_wh, initial)

    def test_battery_never_below_zero(self):
        """Battery energy never goes negative even under extreme drain."""
        cfg = PowerConfig(battery_initial_pct=0.01, battery_capacity_wh=100.0)
        pm = PowerModel(cfg)

        # Drain heavily for a long time
        for _ in range(1000):
            pm.step(60.0, in_eclipse=True, compute_load_w=1000.0)

        self.assertGreaterEqual(pm.battery_wh, 0.0)

    def test_solar_output_zero_during_eclipse(self):
        """Solar panel output is 0 W during eclipse."""
        pm = PowerModel()
        state = pm.step(60.0, in_eclipse=True, compute_load_w=0.0)
        self.assertEqual(state.solar_output_w, 0.0)

    def test_solar_output_positive_in_sunlight(self):
        """Solar panel output is positive in sunlight."""
        pm = PowerModel()
        state = pm.step(60.0, in_eclipse=False, compute_load_w=0.0)
        self.assertGreater(state.solar_output_w, 0.0)

    def test_can_sustain_load_in_sunlight(self):
        """Solar power can sustain moderate load indefinitely in sunlight."""
        pm = PowerModel(PowerConfig(solar_panel_watts=2000, housekeeping_watts=150))
        self.assertTrue(pm.can_sustain_load(500.0, 10.0, in_eclipse=False))

    def test_cannot_sustain_heavy_load_in_eclipse(self):
        """Cannot sustain heavy load in long eclipse with low battery."""
        cfg = PowerConfig(battery_initial_pct=0.25, battery_capacity_wh=1000.0)
        pm = PowerModel(cfg)
        # Heavy load for a long time in eclipse
        self.assertFalse(pm.can_sustain_load(1500.0, 2.0, in_eclipse=True))

    def test_power_state_fields(self):
        """PowerState has all expected fields with sensible values."""
        pm = PowerModel()
        state = pm.step(60.0, in_eclipse=False, compute_load_w=300.0)
        self.assertIsInstance(state, PowerState)
        self.assertGreater(state.battery_pct, 0)
        self.assertLessEqual(state.battery_pct, 1.0)
        self.assertGreaterEqual(state.available_for_compute_w, 0)


# ===================================================================
# thermal.py tests
# ===================================================================
class TestThermal(unittest.TestCase):
    """Tests for thermal model."""

    def test_temperature_rises_with_heat(self):
        """Temperature rises when heat input > radiation output."""
        tm = ThermalModel(initial_temp_c=20.0)
        initial_k = tm.temp_k

        # Large heat input
        tm.step(60.0, heat_load_w=5000.0, in_eclipse=False)

        self.assertGreater(tm.temp_k, initial_k)

    def test_temperature_drops_when_cooling(self):
        """Temperature drops when radiation >> heat input (cold space)."""
        # Start hot, minimal heat input, in eclipse (no solar heating)
        tm = ThermalModel(initial_temp_c=80.0)
        initial_k = tm.temp_k

        tm.step(600.0, heat_load_w=0.0, in_eclipse=True)

        self.assertLess(tm.temp_k, initial_k)

    def test_max_sustainable_heat_positive(self):
        """Max sustainable heat at a moderate temperature is positive."""
        tm = ThermalModel()
        max_heat = tm.max_sustainable_heat_w(target_temp_c=70.0)
        self.assertGreater(max_heat, 0.0)

    def test_max_sustainable_heat_increases_with_temp(self):
        """Higher target temperature allows more heat (T^4 radiation)."""
        tm = ThermalModel()
        h60 = tm.max_sustainable_heat_w(target_temp_c=60.0)
        h80 = tm.max_sustainable_heat_w(target_temp_c=80.0)
        self.assertGreater(h80, h60)

    def test_throttle_activates_above_compute_temp_limit(self):
        """Throttle > 0 when temperature exceeds compute_temp_limit_c."""
        cfg = ThermalConfig(compute_temp_limit_c=75.0, max_temp_c=85.0)
        tm = ThermalModel(cfg, initial_temp_c=80.0)

        state = tm.step(1.0, heat_load_w=0.0, in_eclipse=True)

        self.assertGreater(state.throttle_pct, 0.0)
        self.assertFalse(state.can_compute)

    def test_no_throttle_below_limit(self):
        """No throttle when temperature is well below limit."""
        cfg = ThermalConfig(compute_temp_limit_c=75.0)
        tm = ThermalModel(cfg, initial_temp_c=30.0)

        state = tm.step(1.0, heat_load_w=100.0, in_eclipse=False)

        self.assertEqual(state.throttle_pct, 0.0)
        self.assertTrue(state.can_compute)

    def test_thermal_state_fields(self):
        """ThermalState has correct structure."""
        tm = ThermalModel()
        state = tm.step(60.0, 500.0, in_eclipse=False)
        self.assertIsInstance(state, ThermalState)
        self.assertIsInstance(state.temp_c, float)
        self.assertGreater(state.heat_radiated_w, 0)


# ===================================================================
# scheduler.py tests
# ===================================================================
class TestScheduler(unittest.TestCase):
    """Tests for orbital job scheduler."""

    def test_job_submission_and_queue_ordering(self):
        """Jobs are ordered by priority after submission."""
        sched = OrbitalScheduler()
        sched.submit_job(_simple_job("J-LOW", priority=8))
        sched.submit_job(_simple_job("J-HIGH", priority=2))
        sched.submit_job(_simple_job("J-MED", priority=5))

        priorities = [j.priority for j in sched.job_queue]
        self.assertEqual(priorities, sorted(priorities))
        self.assertEqual(sched.job_queue[0].job_id, "J-HIGH")

    def test_job_starts_when_power_available(self):
        """Scheduler assigns a job when power and battery are sufficient."""
        sched = OrbitalScheduler()
        sched.submit_job(_simple_job("J1", power=300.0))

        decision = sched.decide(
            satellite_name="SAT-000", timestamp=T0,
            power_available_w=500.0, battery_pct=0.80,
            thermal_can_compute=True, thermal_throttle=0.0,
            in_eclipse=False,
        )

        self.assertEqual(decision.action, "run")
        self.assertIsNotNone(decision.job)
        self.assertEqual(decision.job.job_id, "J1")

    def test_job_not_started_insufficient_power(self):
        """Job is not started if power is insufficient."""
        sched = OrbitalScheduler()
        sched.submit_job(_simple_job("J1", power=800.0))

        decision = sched.decide(
            satellite_name="SAT-000", timestamp=T0,
            power_available_w=100.0, battery_pct=0.80,
            thermal_can_compute=True, thermal_throttle=0.0,
            in_eclipse=False,
        )

        self.assertEqual(decision.action, "idle")

    def test_job_preempted_battery_low(self):
        """Running job is preempted when battery drops below min."""
        sched = OrbitalScheduler()
        job = _simple_job("J1", power=300.0, min_battery_pct=0.30)
        sched.submit_job(job)

        # Start the job
        sched.decide("SAT-000", T0, 500.0, 0.80, True, 0.0, False)

        # Now battery drops
        decision = sched.decide(
            "SAT-000", T0 + timedelta(minutes=5),
            500.0, 0.15,  # below min_battery_pct
            True, 0.0, False,
        )

        self.assertEqual(decision.action, "pause")

    def test_completed_jobs_tracked(self):
        """Completed jobs move to completed_jobs list."""
        sched = OrbitalScheduler()
        sched.submit_job(_simple_job("J1", power=300.0, duration=60.0))

        # Start
        sched.decide("SAT-000", T0, 500.0, 0.80, True, 0.0, False)
        # Advance past completion
        sched.advance_job("SAT-000", 120.0, throttle_pct=0.0, timestamp=T0)

        self.assertEqual(len(sched.completed_jobs), 1)
        self.assertEqual(sched.completed_jobs[0].job_id, "J1")
        self.assertEqual(sched.completed_jobs[0].status, JobStatus.COMPLETED)

    def test_stats_accurate(self):
        """Stats reflect the correct counts."""
        sched = OrbitalScheduler()
        sched.submit_job(_simple_job("J1", power=300.0, duration=60.0))
        sched.submit_job(_simple_job("J2", power=300.0, duration=600.0))

        # Start J1
        sched.decide("SAT-000", T0, 500.0, 0.80, True, 0.0, False)
        # Complete J1
        sched.advance_job("SAT-000", 120.0, 0.0, T0)

        stats = sched.stats()
        self.assertEqual(stats["completed"], 1)
        self.assertEqual(stats["queued"], 1)
        self.assertEqual(stats["total_jobs"], 2)

    def test_eclipse_low_battery_causes_charge(self):
        """In eclipse with low battery, scheduler chooses 'charge' over new jobs."""
        sched = OrbitalScheduler()
        sched.submit_job(_simple_job("J1", power=200.0))

        decision = sched.decide(
            "SAT-000", T0,
            power_available_w=500.0, battery_pct=0.35,
            thermal_can_compute=True, thermal_throttle=0.0,
            in_eclipse=True,
        )

        self.assertEqual(decision.action, "charge")

    def test_thermal_limit_causes_idle(self):
        """Scheduler idles when thermal can_compute is False."""
        sched = OrbitalScheduler()
        sched.submit_job(_simple_job("J1", power=200.0))

        decision = sched.decide(
            "SAT-000", T0,
            power_available_w=500.0, battery_pct=0.80,
            thermal_can_compute=False, thermal_throttle=0.8,
            in_eclipse=False,
        )

        self.assertEqual(decision.action, "idle")

    def test_submit_jobs_batch(self):
        """submit_jobs adds multiple jobs in priority order."""
        sched = OrbitalScheduler()
        jobs = [_simple_job(f"J{i}", priority=10 - i) for i in range(5)]
        sched.submit_jobs(jobs)
        self.assertEqual(len(sched.job_queue), 5)
        # Should be sorted by priority
        priorities = [j.priority for j in sched.job_queue]
        self.assertEqual(priorities, sorted(priorities))


# ===================================================================
# ground_stations.py tests
# ===================================================================
class TestGroundStations(unittest.TestCase):
    """Tests for ground station contact window calculation."""

    def test_contact_window_found_for_overhead_pass(self):
        """At least one contact window is found over several orbits."""
        sat = _make_sat()
        windows = find_contact_windows(
            sat, DEFAULT_GROUND_STATIONS,
            T0, duration_hours=6.0, step_seconds=30.0,
        )
        # Over 6 hours with 10 stations, very likely at least one pass
        self.assertGreater(len(windows), 0, "Should find at least one contact window")

        for w in windows:
            self.assertGreater(w.duration_seconds, 0)
            self.assertGreater(w.max_elevation_deg, 0)

    def test_no_contact_opposite_side(self):
        """A ground station on the opposite side of Earth sees no pass
        when the satellite is fixed at one position (short window)."""
        # Use satellite at high latitude
        sat = _make_sat(0)
        # Station at south pole — 53-degree inclination satellite won't reach it
        south_pole = GroundStation("SouthPole", -89.0, 0.0, min_elevation_deg=30.0)

        windows = find_contact_windows(
            sat, [south_pole], T0, duration_hours=2.0, step_seconds=30.0,
        )
        # 53-degree inclination satellite cannot reach 89S at 30 deg elevation
        self.assertEqual(len(windows), 0,
                         "53-deg inclination satellite should not reach South Pole at 30 deg min elevation")

    def test_elevation_angle_directly_overhead(self):
        """Elevation should be ~90 degrees for satellite directly overhead."""
        station = GroundStation("Test", 0.0, 0.0)
        # Station at equator, 0 lon. Put satellite directly above at t=T0.
        station_ecef = _lla_to_ecef(0.0, 0.0)
        station_eci = _ecef_to_eci(station_ecef, T0)
        # Satellite directly above station at 550 km
        up = station_eci / np.linalg.norm(station_eci)
        sat_eci = station_eci + up * 550.0

        elev = elevation_angle(sat_eci, station, T0)
        self.assertGreater(elev, 85.0, "Directly overhead should be near 90 degrees")

    def test_default_stations_count(self):
        """Default ground station list has 10 stations."""
        self.assertEqual(len(DEFAULT_GROUND_STATIONS), 10)


# ===================================================================
# workloads.py tests
# ===================================================================
class TestWorkloads(unittest.TestCase):
    """Tests for workload generation module."""

    def test_all_9_workload_types_valid(self):
        """All 9 catalog entries produce valid ComputeJob instances."""
        self.assertEqual(len(WORKLOAD_CATALOG), 9)

        for key in WORKLOAD_CATALOG:
            job = create_job(key, submit_time=T0)
            self.assertIsInstance(job, ComputeJob)
            self.assertGreater(job.power_watts, 0)
            self.assertGreater(job.duration_seconds, 0)
            self.assertTrue(job.job_id.startswith("WL-"))

    def test_convenience_factories(self):
        """All 9 convenience factory functions produce jobs."""
        factories = [
            image_classification, change_detection, object_tracking,
            llm_inference, image_generation, weather_model,
            climate_analysis, sar_processing, signal_analysis,
        ]
        for fn in factories:
            job = fn(T0)
            self.assertIsInstance(job, ComputeJob)
            self.assertGreater(job.power_watts, 0)

    def test_workload_generator_mix_ratios(self):
        """WorkloadGenerator respects category mix (roughly)."""
        gen = WorkloadGenerator(seed=42)
        jobs = gen.generate_batch(200, T0, duration_hours=6.0)

        self.assertEqual(len(jobs), 200)

        # Count by name to check distribution isn't degenerate
        names = set(j.name for j in jobs)
        self.assertGreater(len(names), 3, "Should have variety of job types")

    def test_workload_generator_summary(self):
        """Summary has expected keys and sensible values."""
        gen = WorkloadGenerator(seed=123)
        jobs = gen.generate_batch(50, T0)
        summary = gen.summary(jobs)

        self.assertEqual(summary["total_jobs"], 50)
        self.assertGreater(summary["total_compute_hours"], 0)
        self.assertGreater(summary["total_energy_kwh"], 0)
        self.assertIn("by_workload", summary)
        self.assertIn("by_type", summary)

    def test_jitter_creates_variation(self):
        """Jobs from same workload type have different power/duration due to jitter."""
        powers = set()
        for _ in range(20):
            job = create_job("llm_inference", T0)
            powers.add(round(job.power_watts, 1))
        self.assertGreater(len(powers), 1, "Jitter should produce varying power values")

    def test_deadline_set_when_submit_time_provided(self):
        """Realtime jobs get a deadline when submit_time is given."""
        job = create_job("signal_analysis", submit_time=T0)
        self.assertIsNotNone(job.deadline)
        self.assertGreater(job.deadline, T0)

    def test_batch_jobs_no_deadline_without_submit_time(self):
        """Batch jobs without submit_time have no deadline."""
        job = create_job("weather_model")
        self.assertIsNone(job.deadline)


if __name__ == "__main__":
    unittest.main()
