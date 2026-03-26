"""Tests for cost_model module — orbital vs terrestrial compute economics."""

import unittest

from orbital_compute.cost_model import (
    ConstellationCostConfig,
    HardwareCosts,
    LaunchCosts,
    OperatingCosts,
    RevenueModel,
    TerrestrialComparison,
    calculate_constellation_costs,
)


class TestLaunchCosts(unittest.TestCase):
    """Tests for launch cost calculations."""

    def test_starship_cheaper_than_falcon9(self):
        """Starship per-kg cost is lower than Falcon 9."""
        lc = LaunchCosts()
        self.assertLess(lc.starship_per_kg, lc.falcon9_rideshare_per_kg)

    def test_launch_cost_scales_with_mass(self):
        """Launch cost = mass_kg * cost_per_kg."""
        lc = LaunchCosts()
        cost_100 = lc.launch_cost(100.0, "falcon9")
        cost_200 = lc.launch_cost(200.0, "falcon9")
        self.assertAlmostEqual(cost_200, cost_100 * 2.0)

    def test_unknown_vehicle_raises(self):
        """Unknown launch vehicle raises ValueError."""
        lc = LaunchCosts()
        with self.assertRaises(ValueError):
            lc.cost_per_kg("saturn_v")

    def test_default_vehicle_used(self):
        """Default vehicle is used when none specified."""
        lc = LaunchCosts(default_vehicle="starship")
        cost = lc.launch_cost(100.0)
        expected = 100.0 * lc.starship_per_kg
        self.assertAlmostEqual(cost, expected)


class TestHardwareCosts(unittest.TestCase):
    """Tests for hardware cost calculations."""

    def test_total_hardware_positive(self):
        """Total hardware cost is positive."""
        hw = HardwareCosts()
        self.assertGreater(hw.total_hardware_cost(), 0.0)

    def test_rad_hardening_increases_electronics_cost(self):
        """Radiation hardening adds to electronics cost."""
        hw = HardwareCosts(radiation_hardening_pct=0.30)
        self.assertGreater(hw.electronics_cost_radhard(), hw.electronics_cost())

    def test_more_gpus_more_cost(self):
        """More GPUs per satellite increases cost."""
        hw1 = HardwareCosts(n_gpus=1)
        hw4 = HardwareCosts(n_gpus=4)
        self.assertGreater(hw4.total_hardware_cost(), hw1.total_hardware_cost())


class TestConstellationCosts(unittest.TestCase):
    """Tests for full constellation cost analysis."""

    def test_capex_increases_with_more_satellites(self):
        """More satellites means higher CAPEX."""
        cfg6 = ConstellationCostConfig(n_satellites=6)
        cfg12 = ConstellationCostConfig(n_satellites=12)
        a6 = calculate_constellation_costs(cfg6, utilization_pct=50.0)
        a12 = calculate_constellation_costs(cfg12, utilization_pct=50.0)
        self.assertGreater(
            a12["capex"]["total_constellation"],
            a6["capex"]["total_constellation"],
        )

    def test_cost_per_hour_decreases_with_utilization(self):
        """Cost per compute-hour decreases with higher utilization."""
        cfg = ConstellationCostConfig(n_satellites=6)
        a_low = calculate_constellation_costs(cfg, utilization_pct=25.0)
        a_high = calculate_constellation_costs(cfg, utilization_pct=75.0)
        self.assertGreater(
            a_low["economics"]["cost_per_compute_hour"],
            a_high["economics"]["cost_per_compute_hour"],
        )

    def test_starship_launch_cheaper_total(self):
        """Using Starship results in lower launch cost than Falcon 9."""
        cfg_f9 = ConstellationCostConfig(
            n_satellites=6,
            launch=LaunchCosts(default_vehicle="falcon9"),
        )
        cfg_ss = ConstellationCostConfig(
            n_satellites=6,
            launch=LaunchCosts(default_vehicle="starship"),
        )
        a_f9 = calculate_constellation_costs(cfg_f9, utilization_pct=50.0)
        a_ss = calculate_constellation_costs(cfg_ss, utilization_pct=50.0)
        # Starship per-sat launch cost should be lower
        self.assertLess(
            a_ss["capex"]["per_satellite"]["launch"],
            a_f9["capex"]["per_satellite"]["launch"],
        )

    def test_breakeven_utilization_valid(self):
        """Breakeven utilization is a positive finite number."""
        cfg = ConstellationCostConfig(n_satellites=6)
        analysis = calculate_constellation_costs(cfg, utilization_pct=50.0)
        be = analysis["economics"]["breakeven_utilization_pct"]
        self.assertGreater(be, 0.0)
        self.assertLess(be, 1000.0)  # Should be a reasonable percentage

    def test_result_has_expected_top_keys(self):
        """Result dict has all expected top-level keys."""
        analysis = calculate_constellation_costs()
        expected = [
            "constellation", "capex", "opex_annual",
            "revenue", "economics", "terrestrial_comparison",
        ]
        for key in expected:
            self.assertIn(key, analysis)

    def test_utilization_sensitivity_has_entries(self):
        """Utilization sensitivity sweep has multiple entries."""
        analysis = calculate_constellation_costs()
        sweep = analysis["economics"]["utilization_sensitivity"]
        self.assertGreater(len(sweep), 3)

    def test_zero_utilization_infinite_cost_per_hour(self):
        """Zero utilization gives infinite cost per hour."""
        cfg = ConstellationCostConfig(n_satellites=6)
        analysis = calculate_constellation_costs(cfg, utilization_pct=0.0)
        self.assertEqual(analysis["economics"]["cost_per_compute_hour"], float("inf"))


class TestTerrestrialComparison(unittest.TestCase):
    """Tests for terrestrial data center comparison."""

    def test_cost_per_hour_positive(self):
        """Terrestrial cost per compute-hour is positive."""
        tc = TerrestrialComparison()
        self.assertGreater(tc.cost_per_compute_hour(50.0), 0.0)

    def test_higher_util_lower_cost_per_hour(self):
        """Higher utilization lowers cost per compute-hour."""
        tc = TerrestrialComparison()
        c50 = tc.cost_per_compute_hour(50.0)
        c90 = tc.cost_per_compute_hour(90.0)
        self.assertGreater(c50, c90)

    def test_orbital_more_expensive_than_terrestrial(self):
        """Orbital compute is more expensive per hour than terrestrial at same util."""
        analysis = calculate_constellation_costs(utilization_pct=50.0)
        self.assertGreater(
            analysis["terrestrial_comparison"]["orbital_cost_multiple"], 1.0,
        )


if __name__ == "__main__":
    unittest.main()
