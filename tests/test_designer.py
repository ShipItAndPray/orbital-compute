"""Tests for designer module — constellation design optimizer."""

import unittest

from orbital_compute.designer import (
    ConstellationDesign,
    ConstellationDesigner,
    DesignRequirements,
    eclipse_fraction_estimate,
    max_altitude_for_latency,
    orbital_period_s,
    swath_radius_km,
)


class TestOrbitalHelpers(unittest.TestCase):
    """Tests for orbital mechanics helper functions."""

    def test_orbital_period_increases_with_altitude(self):
        """Higher altitude means longer orbital period."""
        p400 = orbital_period_s(400.0)
        p800 = orbital_period_s(800.0)
        self.assertGreater(p800, p400)

    def test_orbital_period_leo_range(self):
        """LEO orbital period is roughly 90-100 minutes."""
        period = orbital_period_s(550.0)
        minutes = period / 60.0
        self.assertGreater(minutes, 85.0)
        self.assertLess(minutes, 105.0)

    def test_eclipse_fraction_between_zero_and_one(self):
        """Eclipse fraction is always in (0, 1)."""
        for alt in [400, 550, 800, 1200]:
            for inc in [0, 30, 53, 70, 90, 97]:
                ef = eclipse_fraction_estimate(float(alt), float(inc))
                self.assertGreater(ef, 0.0)
                self.assertLess(ef, 1.0)

    def test_higher_altitude_lower_eclipse(self):
        """Higher altitude generally means slightly lower eclipse fraction
        (Earth subtends smaller angle)."""
        ef_low = eclipse_fraction_estimate(400.0, 53.0)
        ef_high = eclipse_fraction_estimate(1200.0, 53.0)
        self.assertGreater(ef_low, ef_high)

    def test_sun_sync_low_eclipse(self):
        """Sun-synchronous orbits (96-99 deg) have lower eclipse fraction."""
        ef_sso = eclipse_fraction_estimate(550.0, 97.0)
        ef_normal = eclipse_fraction_estimate(550.0, 53.0)
        self.assertLess(ef_sso, ef_normal)

    def test_max_altitude_for_latency(self):
        """Reasonable latency gives reasonable altitude constraint."""
        alt = max_altitude_for_latency(50.0)  # 50ms
        self.assertGreater(alt, 0.0)
        self.assertLess(alt, 10000.0)  # Should not be unreasonable

    def test_swath_radius_positive(self):
        """Swath radius is positive at LEO."""
        sr = swath_radius_km(550.0)
        self.assertGreater(sr, 0.0)

    def test_swath_increases_with_altitude(self):
        """Higher altitude means larger ground swath."""
        sr_low = swath_radius_km(400.0)
        sr_high = swath_radius_km(800.0)
        self.assertGreater(sr_high, sr_low)


class TestConstellationDesigner(unittest.TestCase):
    """Tests for the ConstellationDesigner optimizer."""

    def test_returns_valid_design_within_budget(self):
        """Designer returns a design with cost <= budget."""
        req = DesignRequirements(
            target_coverage="global",
            compute_capacity_gpu_hours_day=100.0,
            budget_usd=50_000_000.0,
            max_latency_ms=50.0,
        )
        designer = ConstellationDesigner(req)
        design = designer.design()
        self.assertIsInstance(design, ConstellationDesign)
        if design.n_satellites > 0:
            # Budget with 10% slack
            self.assertLessEqual(design.estimated_cost_usd,
                                 req.budget_usd * 1.1)

    def test_more_compute_more_satellites(self):
        """Requesting more compute generally means more satellites or GPUs."""
        req_small = DesignRequirements(
            compute_capacity_gpu_hours_day=50.0,
            budget_usd=100_000_000.0,
        )
        req_large = DesignRequirements(
            compute_capacity_gpu_hours_day=500.0,
            budget_usd=100_000_000.0,
        )
        d_small = ConstellationDesigner(req_small).design()
        d_large = ConstellationDesigner(req_large).design()
        # More compute should need more total GPUs
        self.assertGreaterEqual(d_large.total_gpus, d_small.total_gpus)

    def test_design_has_positive_satellites(self):
        """With reasonable budget, design has at least 1 satellite."""
        req = DesignRequirements(
            compute_capacity_gpu_hours_day=24.0,
            budget_usd=20_000_000.0,
        )
        design = ConstellationDesigner(req).design()
        self.assertGreater(design.n_satellites, 0)

    def test_design_has_walker_notation(self):
        """Design output includes Walker notation string."""
        req = DesignRequirements(
            compute_capacity_gpu_hours_day=100.0,
            budget_usd=50_000_000.0,
        )
        design = ConstellationDesigner(req).design()
        if design.n_satellites > 0:
            self.assertIn(":", design.walker_notation)
            self.assertIn("/", design.walker_notation)

    def test_alternatives_considered_positive(self):
        """Designer evaluates multiple design alternatives."""
        req = DesignRequirements(
            compute_capacity_gpu_hours_day=100.0,
            budget_usd=50_000_000.0,
        )
        design = ConstellationDesigner(req).design()
        self.assertGreater(design.alternatives_considered, 0)

    def test_insufficient_budget_produces_fallback(self):
        """Impossibly low budget returns a design with warnings."""
        req = DesignRequirements(
            compute_capacity_gpu_hours_day=10000.0,
            budget_usd=100.0,  # impossibly low
        )
        design = ConstellationDesigner(req).design()
        # Should have design notes about infeasibility
        self.assertGreater(len(design.design_notes), 0)

    def test_invalid_coverage_raises(self):
        """Invalid coverage type raises ValueError."""
        with self.assertRaises(ValueError):
            DesignRequirements(target_coverage="mars_orbit")

    def test_eclipse_fraction_in_design(self):
        """Design includes eclipse fraction within constraint."""
        req = DesignRequirements(
            compute_capacity_gpu_hours_day=100.0,
            budget_usd=50_000_000.0,
            max_eclipse_fraction=0.40,
        )
        design = ConstellationDesigner(req).design()
        if design.n_satellites > 0:
            self.assertLessEqual(design.eclipse_fraction, req.max_eclipse_fraction)

    def test_tropical_coverage_low_inclination(self):
        """Tropical coverage uses lower inclination than global."""
        req_tropical = DesignRequirements(
            target_coverage="tropical",
            compute_capacity_gpu_hours_day=100.0,
            budget_usd=50_000_000.0,
        )
        req_global = DesignRequirements(
            target_coverage="global",
            compute_capacity_gpu_hours_day=100.0,
            budget_usd=50_000_000.0,
        )
        d_trop = ConstellationDesigner(req_tropical).design()
        d_glob = ConstellationDesigner(req_global).design()
        if d_trop.n_satellites > 0 and d_glob.n_satellites > 0:
            self.assertLessEqual(d_trop.inclination_deg, d_glob.inclination_deg)


if __name__ == "__main__":
    unittest.main()
