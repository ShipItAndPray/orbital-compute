"""Tests for REST API data accessors."""
import unittest
from orbital_compute.api import SimulationServer
from orbital_compute.simulator import Simulation, SimulationConfig


class TestAPIAccessors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = SimulationConfig(n_satellites=3, sim_duration_hours=1, n_jobs=5)
        cls.sim = Simulation(config)
        cls.sim.setup()
        cls.sim.run()
        cls.server = SimulationServer(cls.sim)

    def test_get_status_has_keys(self):
        status = self.server.get_status()
        self.assertIn("scheduler", status)
        self.assertIn("satellites", status)
        self.assertIn("time", status)

    def test_get_status_satellite_count(self):
        status = self.server.get_status()
        self.assertEqual(len(status["satellites"]), 3)

    def test_get_status_scheduler_completed(self):
        status = self.server.get_status()
        self.assertEqual(status["scheduler"]["completed"], 5)

    def test_get_satellites_returns_list(self):
        sats = self.server.get_satellites()
        self.assertIsInstance(sats, list)
        self.assertEqual(len(sats), 3)

    def test_get_satellites_has_name(self):
        sats = self.server.get_satellites()
        for s in sats:
            self.assertIn("name", s)

    def test_get_jobs_returns_dict(self):
        jobs = self.server.get_jobs()
        self.assertIsInstance(jobs, dict)
        self.assertIn("completed", jobs)

    def test_get_metrics_returns_dict(self):
        metrics = self.server.get_metrics()
        self.assertIsInstance(metrics, dict)

    def test_get_contacts_returns_list(self):
        contacts = self.server.get_contacts()
        self.assertIsInstance(contacts, list)


if __name__ == "__main__":
    unittest.main()
