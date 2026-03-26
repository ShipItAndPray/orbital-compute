"""Tests for reliability module — constellation availability and degradation."""

import math
import unittest

from orbital_compute.reliability import (
    ComponentReliability,
    ReliabilityAnalyzer,
)


class TestComponentReliability(unittest.TestCase):
    """Tests for component reliability calculations."""

    def test_satellite_mtbf_less_than_weakest_component(self):
        """Series system MTBF is always less than the weakest component."""
        rel = ComponentReliability()
        # GPU has the lowest MTBF at 50000 hours
        self.assertLess(rel.satellite_mtbf_hours, rel.gpu_mtbf_hours)

    def test_satellite_mtbf_years_conversion(self):
        """MTBF in years = MTBF in hours / 8760."""
        rel = ComponentReliability()
        self.assertAlmostEqual(
            rel.satellite_mtbf_years,
            rel.satellite_mtbf_hours / 8760.0,
        )


class TestSingleSatAvailability(unittest.TestCase):
    """Tests for single satellite availability."""

    def test_availability_is_mtbf_over_mtbf_plus_mttr(self):
        """Single sat availability = MTBF / (MTBF + MTTR)."""
        analyzer = ReliabilityAnalyzer(n_satellites=1, mttr_hours=8760.0)
        p = analyzer.single_satellite_availability()
        mtbf = analyzer.rel.satellite_mtbf_hours
        expected = mtbf / (mtbf + 8760.0)
        self.assertAlmostEqual(p, expected, places=6)

    def test_availability_between_zero_and_one(self):
        """Availability is always in [0, 1]."""
        analyzer = ReliabilityAnalyzer(n_satellites=1)
        p = analyzer.single_satellite_availability()
        self.assertGreater(p, 0.0)
        self.assertLess(p, 1.0)

    def test_higher_mttr_lowers_availability(self):
        """Longer repair time reduces availability."""
        a_fast = ReliabilityAnalyzer(n_satellites=1, mttr_hours=1000.0)
        a_slow = ReliabilityAnalyzer(n_satellites=1, mttr_hours=20000.0)
        self.assertGreater(
            a_fast.single_satellite_availability(),
            a_slow.single_satellite_availability(),
        )


class TestConstellationAvailability(unittest.TestCase):
    """Tests for constellation-level availability."""

    def test_single_sat_required_approx_1_minus_1_minus_p_to_n(self):
        """With min_operational=1, availability ~ 1 - (1-p)^n."""
        n = 6
        analyzer = ReliabilityAnalyzer(n_satellites=n)
        result = analyzer.constellation_availability(min_operational=1)
        p = analyzer.single_satellite_availability()
        expected = 1.0 - (1.0 - p) ** n
        self.assertAlmostEqual(result.constellation_availability, expected, places=6)

    def test_full_constellation_lower_than_single_sat(self):
        """Requiring all sats operational has lower availability than requiring just 1."""
        n = 6
        analyzer = ReliabilityAnalyzer(n_satellites=n)
        full = analyzer.constellation_availability(min_operational=n)
        partial = analyzer.constellation_availability(min_operational=1)
        self.assertLess(full.constellation_availability, partial.constellation_availability)

    def test_more_spare_sats_increases_availability(self):
        """Requiring fewer sats yields higher availability."""
        analyzer = ReliabilityAnalyzer(n_satellites=12)
        avail_6 = analyzer.constellation_availability(min_operational=6)
        avail_10 = analyzer.constellation_availability(min_operational=10)
        self.assertGreater(avail_6.constellation_availability,
                           avail_10.constellation_availability)

    def test_expected_operational_is_n_times_p(self):
        """Expected operational satellites = n * p."""
        n = 12
        analyzer = ReliabilityAnalyzer(n_satellites=n)
        result = analyzer.constellation_availability(min_operational=1)
        p = analyzer.single_satellite_availability()
        self.assertAlmostEqual(result.expected_operational, n * p, places=4)

    def test_annual_downtime_nonnegative(self):
        """Annual downtime hours is non-negative."""
        analyzer = ReliabilityAnalyzer(n_satellites=6)
        result = analyzer.constellation_availability(min_operational=3)
        self.assertGreaterEqual(result.annual_downtime_hours, 0.0)

    def test_nines_increases_with_redundancy(self):
        """More redundancy means more nines of availability."""
        analyzer = ReliabilityAnalyzer(n_satellites=12)
        avail_1 = analyzer.constellation_availability(min_operational=1)
        avail_6 = analyzer.constellation_availability(min_operational=6)
        self.assertGreater(avail_1.nines, avail_6.nines)


class TestDegradationCurve(unittest.TestCase):
    """Tests for degradation curve."""

    def test_probabilities_sum_to_approximately_one(self):
        """Probabilities over all states should sum to ~1.0."""
        analyzer = ReliabilityAnalyzer(n_satellites=6)
        curve = analyzer.degradation_curve()
        total_prob = sum(entry["probability"] for entry in curve)
        self.assertAlmostEqual(total_prob, 1.0, places=2)  # Degradation curve omits 0-operational

    def test_curve_length_equals_n_satellites(self):
        """Curve has one entry per possible operational count (n down to 1)."""
        n = 8
        analyzer = ReliabilityAnalyzer(n_satellites=n)
        curve = analyzer.degradation_curve()
        self.assertEqual(len(curve), n)

    def test_capacity_pct_decreasing(self):
        """Capacity percentage decreases along the curve."""
        analyzer = ReliabilityAnalyzer(n_satellites=6)
        curve = analyzer.degradation_curve()
        capacities = [e["capacity_pct"] for e in curve]
        self.assertEqual(capacities, sorted(capacities, reverse=True))

    def test_cumulative_availability_monotonic(self):
        """Cumulative availability changes monotonically across degradation curve."""
        analyzer = ReliabilityAnalyzer(n_satellites=6)
        curve = analyzer.degradation_curve()
        # Curve goes from n_sats (hard) to 1 (easy), so cumulative increases
        cum_avails = [e["cumulative_availability"] for e in curve]
        for i in range(len(cum_avails) - 1):
            self.assertLessEqual(cum_avails[i], cum_avails[i + 1] + 1e-10)


class TestSLAAnalysis(unittest.TestCase):
    """Tests for SLA analysis."""

    def test_sla_returns_all_tiers(self):
        """SLA analysis returns results for all 4 standard tiers."""
        analyzer = ReliabilityAnalyzer(n_satellites=12)
        sla = analyzer.sla_analysis()
        expected_tiers = ["99%", "99.9%", "99.99%", "99.999%"]
        for tier in expected_tiers:
            self.assertIn(tier, sla)

    def test_sla_achievability_has_correct_keys(self):
        """Each SLA tier result has required keys."""
        analyzer = ReliabilityAnalyzer(n_satellites=12)
        sla = analyzer.sla_analysis()
        for tier, data in sla.items():
            self.assertIn("achievable", data)
            self.assertIn("min_operational", data)
            self.assertIn("spare_satellites", data)
            self.assertIn("actual_availability", data)
            self.assertIn("annual_downtime_hours", data)

    def test_lower_sla_easier_to_achieve(self):
        """99% SLA is achievable if 99.9% is achievable (monotonicity)."""
        analyzer = ReliabilityAnalyzer(n_satellites=12)
        sla = analyzer.sla_analysis()
        if sla["99.9%"]["achievable"]:
            self.assertTrue(sla["99%"]["achievable"])


if __name__ == "__main__":
    unittest.main()
