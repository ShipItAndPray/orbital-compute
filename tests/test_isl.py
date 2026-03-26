"""Tests for ISL module — inter-satellite link simulation."""

import unittest

import numpy as np

from orbital_compute.orbit import EARTH_RADIUS_KM
from orbital_compute.isl import (
    has_line_of_sight,
    link_metrics,
    MAX_LINK_RANGE_KM,
    FULL_BW_RANGE_KM,
    MAX_BANDWIDTH_GBPS,
    SPEED_OF_LIGHT_KM_S,
)


class TestLineOfSight(unittest.TestCase):
    """Tests for line-of-sight calculation."""

    def test_opposite_side_blocked_by_earth(self):
        """Satellites on opposite sides of Earth cannot see each other."""
        alt = 550.0
        sat1 = np.array([EARTH_RADIUS_KM + alt, 0.0, 0.0])
        sat2 = np.array([-(EARTH_RADIUS_KM + alt), 0.0, 0.0])
        self.assertFalse(has_line_of_sight(sat1, sat2))

    def test_nearby_sats_have_los(self):
        """Nearby satellites in similar orbit have line of sight."""
        alt = 550.0
        sat1 = np.array([EARTH_RADIUS_KM + alt, 0.0, 0.0])
        sat2 = np.array([EARTH_RADIUS_KM + alt, 100.0, 0.0])
        self.assertTrue(has_line_of_sight(sat1, sat2))

    def test_same_position_trivially_visible(self):
        """Same position is trivially visible."""
        pos = np.array([EARTH_RADIUS_KM + 550, 0.0, 0.0])
        self.assertTrue(has_line_of_sight(pos, pos))

    def test_sats_above_horizon_visible(self):
        """Two satellites above Earth with no Earth blockage have LOS."""
        alt = 550.0
        r = EARTH_RADIUS_KM + alt
        # Small angular separation — both on the same hemisphere
        angle_rad = np.radians(10.0)
        sat1 = np.array([r, 0.0, 0.0])
        sat2 = np.array([r * np.cos(angle_rad), r * np.sin(angle_rad), 0.0])
        self.assertTrue(has_line_of_sight(sat1, sat2))

    def test_sats_on_edge_of_earth_shadow(self):
        """Satellites separated by ~120 degrees may be blocked by Earth."""
        alt = 550.0
        r = EARTH_RADIUS_KM + alt
        angle_rad = np.radians(120.0)
        sat1 = np.array([r, 0.0, 0.0])
        sat2 = np.array([r * np.cos(angle_rad), r * np.sin(angle_rad), 0.0])
        # At 120 degrees separation at LEO, the line passes through Earth
        self.assertFalse(has_line_of_sight(sat1, sat2))


class TestLinkMetrics(unittest.TestCase):
    """Tests for link bandwidth, latency, and range."""

    def test_max_range_enforced(self):
        """Links beyond MAX_LINK_RANGE_KM are not in range."""
        sat1 = np.array([EARTH_RADIUS_KM + 550, 0.0, 0.0])
        # Place sat2 far away (beyond 5000 km)
        sat2 = np.array([EARTH_RADIUS_KM + 550, 6000.0, 0.0])
        dist = np.linalg.norm(sat2 - sat1)
        self.assertGreater(dist, MAX_LINK_RANGE_KM)
        metrics = link_metrics(sat1, sat2)
        self.assertFalse(metrics.in_range)
        self.assertAlmostEqual(metrics.bandwidth_gbps, 0.0)

    def test_full_bandwidth_within_1000km(self):
        """Links within FULL_BW_RANGE_KM get full bandwidth."""
        sat1 = np.array([EARTH_RADIUS_KM + 550, 0.0, 0.0])
        sat2 = np.array([EARTH_RADIUS_KM + 550, 500.0, 0.0])
        dist = np.linalg.norm(sat2 - sat1)
        self.assertLess(dist, FULL_BW_RANGE_KM)
        metrics = link_metrics(sat1, sat2)
        self.assertTrue(metrics.in_range)
        self.assertAlmostEqual(metrics.bandwidth_gbps, MAX_BANDWIDTH_GBPS)

    def test_bandwidth_degrades_with_distance(self):
        """Bandwidth at 3000 km is less than at 500 km."""
        sat1 = np.array([EARTH_RADIUS_KM + 550, 0.0, 0.0])
        sat_near = np.array([EARTH_RADIUS_KM + 550, 500.0, 0.0])
        sat_far = np.array([EARTH_RADIUS_KM + 550, 3000.0, 0.0])
        m_near = link_metrics(sat1, sat_near)
        m_far = link_metrics(sat1, sat_far)
        if m_far.in_range:
            self.assertLess(m_far.bandwidth_gbps, m_near.bandwidth_gbps)

    def test_latency_proportional_to_distance(self):
        """Latency = distance / speed_of_light."""
        sat1 = np.array([EARTH_RADIUS_KM + 550, 0.0, 0.0])
        sat2 = np.array([EARTH_RADIUS_KM + 550, 1000.0, 0.0])
        metrics = link_metrics(sat1, sat2)
        expected_latency = metrics.distance_km / SPEED_OF_LIGHT_KM_S
        self.assertAlmostEqual(metrics.latency_s, expected_latency, places=6)

    def test_latency_positive(self):
        """Latency is always positive for distinct positions."""
        sat1 = np.array([EARTH_RADIUS_KM + 550, 0.0, 0.0])
        sat2 = np.array([EARTH_RADIUS_KM + 550, 100.0, 0.0])
        metrics = link_metrics(sat1, sat2)
        self.assertGreater(metrics.latency_s, 0.0)

    def test_bandwidth_at_max_range_is_zero(self):
        """At exactly MAX_LINK_RANGE_KM, bandwidth should be ~0 (edge case)."""
        sat1 = np.array([0.0, 0.0, 0.0])
        sat2 = np.array([MAX_LINK_RANGE_KM, 0.0, 0.0])
        metrics = link_metrics(sat1, sat2)
        # At exactly max range, fraction = 1 - (5000-1000)/(5000-1000) = 0
        self.assertAlmostEqual(metrics.bandwidth_gbps, 0.0, places=1)


class TestLinkBandwidthModel(unittest.TestCase):
    """Tests for the linear bandwidth degradation model."""

    def test_midpoint_bandwidth(self):
        """At midpoint between FULL_BW and MAX range, bandwidth is ~half."""
        midpoint_dist = (FULL_BW_RANGE_KM + MAX_LINK_RANGE_KM) / 2.0
        sat1 = np.array([0.0, 0.0, 0.0])
        sat2 = np.array([midpoint_dist, 0.0, 0.0])
        metrics = link_metrics(sat1, sat2)
        expected_fraction = 0.5
        expected_bw = MAX_BANDWIDTH_GBPS * expected_fraction
        self.assertAlmostEqual(metrics.bandwidth_gbps, expected_bw, places=1)


if __name__ == "__main__":
    unittest.main()
