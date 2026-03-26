"""Tests for the look-ahead scheduler (v2)."""
import unittest
from datetime import datetime, timezone, timedelta
from orbital_compute.scheduler_v2 import LookAheadScheduler
from orbital_compute.scheduler import ComputeJob, JobType, JobStatus


class TestLookAheadScheduler(unittest.TestCase):
    def setUp(self):
        self.scheduler = LookAheadScheduler()
        self.t = datetime(2026, 3, 26, 12, 0, 0, tzinfo=timezone.utc)

    def test_submit_and_decide_sunlit(self):
        job = ComputeJob("J1", "test", power_watts=200, duration_seconds=300)
        self.scheduler.submit_job(job)
        decision = self.scheduler.decide("SAT-0", self.t, power_available_w=1000,
                                          battery_pct=0.8, thermal_can_compute=True,
                                          thermal_throttle=0.0, in_eclipse=False)
        self.assertEqual(decision.action, "run")

    def test_eclipse_low_battery_charges(self):
        job = ComputeJob("J1", "test", power_watts=200, duration_seconds=300)
        self.scheduler.submit_job(job)
        decision = self.scheduler.decide("SAT-0", self.t, power_available_w=500,
                                          battery_pct=0.3, thermal_can_compute=True,
                                          thermal_throttle=0.0, in_eclipse=True)
        self.assertEqual(decision.action, "charge")

    def test_preempt_on_low_battery(self):
        job = ComputeJob("J1", "test", power_watts=200, duration_seconds=300,
                          min_battery_pct=0.4, checkpointable=True)
        self.scheduler.submit_job(job)
        # Start job
        self.scheduler.decide("SAT-0", self.t, power_available_w=1000,
                               battery_pct=0.8, thermal_can_compute=True,
                               thermal_throttle=0.0, in_eclipse=False)
        # Battery drops below job minimum
        decision = self.scheduler.decide("SAT-0", self.t + timedelta(minutes=1),
                                          power_available_w=500, battery_pct=0.2,
                                          thermal_can_compute=True, thermal_throttle=0.0,
                                          in_eclipse=True)
        self.assertEqual(decision.action, "pause")

    def test_eclipse_forecast_used(self):
        """Scheduler uses eclipse forecast for scoring."""
        eclipse_start = self.t + timedelta(minutes=10)
        eclipse_end = self.t + timedelta(minutes=45)
        self.scheduler.set_eclipse_forecast("SAT-0", [(eclipse_start, eclipse_end)])
        remaining = self.scheduler._sunlit_remaining("SAT-0", self.t)
        self.assertAlmostEqual(remaining, 600.0, places=0)  # 10 min = 600s

    def test_load_balancing_tracks_completions(self):
        job = ComputeJob("J1", "test", power_watts=100, duration_seconds=1)
        self.scheduler.submit_job(job)
        self.scheduler.decide("SAT-0", self.t, power_available_w=1000,
                               battery_pct=0.9, thermal_can_compute=True,
                               thermal_throttle=0.0, in_eclipse=False)
        self.scheduler.advance_job("SAT-0", 10.0, 0.0, self.t)
        self.assertEqual(self.scheduler.sat_job_counts.get("SAT-0", 0), 1)


if __name__ == "__main__":
    unittest.main()
