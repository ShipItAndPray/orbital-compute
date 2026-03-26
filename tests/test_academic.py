"""Tests for academic algorithm implementations (PHOENIX, CGR)."""
import unittest
from datetime import datetime, timezone
from orbital_compute.phoenix import PhoenixScheduler, OrbitAwareScheduler
from orbital_compute.scheduler import ComputeJob
from orbital_compute.cgr import ContactPlanGenerator, CGRRouter, Contact
from orbital_compute.orbit import starlink_shell_1_sample
from orbital_compute.ground_stations import DEFAULT_GROUND_STATIONS


class TestPhoenixScheduler(unittest.TestCase):
    def setUp(self):
        self.t = datetime(2026, 3, 26, 12, 0, 0, tzinfo=timezone.utc)

    def test_submit_and_decide(self):
        sched = PhoenixScheduler()
        job = ComputeJob("J1", "test", power_watts=200, duration_seconds=300)
        sched.submit_job(job)
        decision = sched.decide("SAT-0", self.t, power_available_w=1000,
                                 battery_pct=0.8, thermal_can_compute=True,
                                 thermal_throttle=0.0, in_eclipse=False)
        self.assertEqual(decision.action, "run")

    def test_eclipse_preserves_battery(self):
        sched = PhoenixScheduler()
        job = ComputeJob("J1", "test", power_watts=200, duration_seconds=300)
        sched.submit_job(job)
        decision = sched.decide("SAT-0", self.t, power_available_w=500,
                                 battery_pct=0.3, thermal_can_compute=True,
                                 thermal_throttle=0.0, in_eclipse=True)
        # PHOENIX should protect battery in eclipse
        self.assertIn(decision.action, ["charge", "idle"])


class TestOrbitAwareScheduler(unittest.TestCase):
    def test_basic_scheduling(self):
        sched = OrbitAwareScheduler()
        job = ComputeJob("J1", "test", power_watts=200, duration_seconds=60)
        sched.submit_job(job)
        t = datetime(2026, 3, 26, 12, 0, 0, tzinfo=timezone.utc)
        decision = sched.decide("SAT-0", t, power_available_w=1000,
                                 battery_pct=0.9, thermal_can_compute=True,
                                 thermal_throttle=0.0, in_eclipse=False)
        self.assertEqual(decision.action, "run")


class TestCGRContactPlan(unittest.TestCase):
    def test_generates_contacts(self):
        sats = starlink_shell_1_sample(4)
        stations = DEFAULT_GROUND_STATIONS[:2]
        gen = ContactPlanGenerator(sats, stations)
        t = datetime(2026, 3, 26, 12, 0, 0, tzinfo=timezone.utc)
        contacts = gen.generate(t, duration_hours=0.5)
        self.assertGreater(len(contacts.contacts), 0)

    def test_contacts_have_required_fields(self):
        sats = starlink_shell_1_sample(3)
        gen = ContactPlanGenerator(sats, DEFAULT_GROUND_STATIONS[:1])
        t = datetime(2026, 3, 26, 12, 0, 0, tzinfo=timezone.utc)
        contacts = gen.generate(t, duration_hours=0.5)
        if contacts.contacts:
            c = contacts.contacts[0]
            self.assertIsNotNone(c.from_node)
            self.assertIsNotNone(c.to_node)
            self.assertGreater(c.data_rate_bps, 0)


if __name__ == "__main__":
    unittest.main()
