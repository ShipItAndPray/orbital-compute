"""Tests for K8s scheduler extender."""
import unittest
from orbital_compute.k8s_scheduler import (
    OrbitalSchedulerExtender, PodOrbitalConstraints,
    SatelliteState, ConstellationState,
)


class TestPodConstraints(unittest.TestCase):
    def test_default_constraints(self):
        c = PodOrbitalConstraints.from_annotations({})
        self.assertEqual(c.min_battery_pct, 20.0)
        self.assertTrue(c.prefer_sunlit)

    def test_custom_annotations(self):
        ann = {
            "orbital-compute/min-battery": "50",
            "orbital-compute/prefer-sunlit": "false",
            "orbital-compute/power-watts": "700",
        }
        c = PodOrbitalConstraints.from_annotations(ann)
        self.assertEqual(c.min_battery_pct, 50.0)
        self.assertFalse(c.prefer_sunlit)
        self.assertEqual(c.power_watts, 700.0)


class TestExtenderFilter(unittest.TestCase):
    def setUp(self):
        state = ConstellationState(satellites={
            "sat-good": SatelliteState(name="sat-good", battery_pct=90, temp_c=35,
                                        available_power_w=1500, time_to_eclipse_min=45),
            "sat-low-batt": SatelliteState(name="sat-low-batt", battery_pct=10, temp_c=25,
                                            in_eclipse=True, available_power_w=300),
        })
        self.extender = OrbitalSchedulerExtender(state)

    def test_filter_removes_low_battery(self):
        pod = {"metadata": {"annotations": {"orbital-compute/min-battery": "30"}}}
        result = self.extender.filter(pod, ["sat-good", "sat-low-batt"])
        passed = [n["metadata"]["name"] for n in result.get("Nodes", {}).get("items", [])]
        self.assertIn("sat-good", passed)
        self.assertNotIn("sat-low-batt", passed)


class TestExtenderPrioritize(unittest.TestCase):
    def setUp(self):
        state = ConstellationState(satellites={
            "sat-sunlit": SatelliteState(name="sat-sunlit", battery_pct=90, temp_c=35,
                                          available_power_w=1500, time_to_eclipse_min=50),
            "sat-eclipse": SatelliteState(name="sat-eclipse", battery_pct=60, temp_c=28,
                                           in_eclipse=True, available_power_w=800),
        })
        self.extender = OrbitalSchedulerExtender(state)

    def test_sunlit_scores_higher(self):
        pod = {"metadata": {"annotations": {"orbital-compute/prefer-sunlit": "true"}}}
        result = self.extender.prioritize(pod, ["sat-sunlit", "sat-eclipse"])
        scores = {e["Host"]: e["Score"] for e in result}
        self.assertGreater(scores["sat-sunlit"], scores["sat-eclipse"])


if __name__ == "__main__":
    unittest.main()
