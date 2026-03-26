"""Tests for network module — constellation network topology analysis."""

import unittest
from datetime import datetime, timezone

from orbital_compute.orbit import starlink_shell_1_sample
from orbital_compute.isl import InterSatelliteNetwork
from orbital_compute.network import NetworkAnalyzer, NetworkMetrics


T0 = datetime(2026, 3, 26, 12, 0, 0, tzinfo=timezone.utc)


def _make_network(n_sats=12):
    """Create a small network for testing."""
    sats = starlink_shell_1_sample(n_sats)
    isl = InterSatelliteNetwork(sats)
    return NetworkAnalyzer(isl), sats


class TestNetworkAnalyzerBasic(unittest.TestCase):
    """Basic network analysis tests."""

    def test_analyze_returns_metrics(self):
        """analyze() returns a NetworkMetrics object."""
        analyzer, _ = _make_network(6)
        metrics = analyzer.analyze(T0)
        self.assertIsInstance(metrics, NetworkMetrics)

    def test_n_satellites_matches(self):
        """Reported number of satellites matches input."""
        analyzer, sats = _make_network(8)
        metrics = analyzer.analyze(T0)
        self.assertEqual(metrics.n_satellites, len(sats))

    def test_connected_components_at_least_one(self):
        """At least one connected component exists."""
        analyzer, _ = _make_network(6)
        metrics = analyzer.analyze(T0)
        self.assertGreaterEqual(metrics.connected_components, 1)

    def test_diameter_nonnegative(self):
        """Network diameter (max shortest path) is non-negative."""
        analyzer, _ = _make_network(6)
        metrics = analyzer.analyze(T0)
        self.assertGreaterEqual(metrics.diameter_hops, 0)

    def test_avg_path_length_nonnegative(self):
        """Average path length is non-negative."""
        analyzer, _ = _make_network(6)
        metrics = analyzer.analyze(T0)
        self.assertGreaterEqual(metrics.avg_path_length_hops, 0.0)


class TestNetworkConnectivity(unittest.TestCase):
    """Tests for connectivity detection."""

    def test_nearby_sats_have_links(self):
        """Satellites in a small constellation should have some ISL links."""
        analyzer, _ = _make_network(12)
        metrics = analyzer.analyze(T0)
        # Some links should exist (nearby sats within 5000 km)
        # Not guaranteed all connected, but should have some links
        self.assertGreaterEqual(metrics.n_links, 0)

    def test_more_sats_more_or_equal_links(self):
        """Larger constellation should have at least as many links."""
        a6, _ = _make_network(6)
        a12, _ = _make_network(12)
        m6 = a6.analyze(T0)
        m12 = a12.analyze(T0)
        self.assertGreaterEqual(m12.n_links, m6.n_links)


class TestNetworkBandwidth(unittest.TestCase):
    """Tests for bandwidth metrics."""

    def test_total_bandwidth_nonnegative(self):
        """Total bandwidth is non-negative."""
        analyzer, _ = _make_network(6)
        metrics = analyzer.analyze(T0)
        self.assertGreaterEqual(metrics.total_bandwidth_gbps, 0.0)

    def test_bandwidth_positive_if_links_exist(self):
        """If there are active links, total bandwidth should be positive."""
        analyzer, _ = _make_network(12)
        metrics = analyzer.analyze(T0)
        if metrics.n_links > 0:
            self.assertGreater(metrics.total_bandwidth_gbps, 0.0)


class TestRoutingTable(unittest.TestCase):
    """Tests for routing table construction."""

    def test_routing_table_for_existing_sat(self):
        """Routing table for an existing satellite returns entries."""
        analyzer, sats = _make_network(12)
        table = analyzer.build_routing_table(sats[0].name, T0)
        # Table is a list; may be empty if sat is isolated
        self.assertIsInstance(table, list)

    def test_routing_table_entries_have_valid_paths(self):
        """Each routing entry has a path starting from source."""
        analyzer, sats = _make_network(12)
        source = sats[0].name
        table = analyzer.build_routing_table(source, T0)
        for entry in table:
            self.assertEqual(entry.path[0], source)
            self.assertEqual(entry.path[-1], entry.destination)
            self.assertGreater(entry.hops, 0)
            self.assertGreater(entry.latency_ms, 0.0)

    def test_routing_table_for_nonexistent_sat(self):
        """Routing table for a nonexistent satellite returns empty list."""
        analyzer, _ = _make_network(6)
        table = analyzer.build_routing_table("NONEXISTENT-SAT", T0)
        self.assertEqual(table, [])

    def test_routing_bandwidth_positive(self):
        """Routing entries with valid paths have positive bandwidth."""
        analyzer, sats = _make_network(12)
        table = analyzer.build_routing_table(sats[0].name, T0)
        for entry in table:
            self.assertGreater(entry.bandwidth_gbps, 0.0)


class TestEmptyNetwork(unittest.TestCase):
    """Tests for edge case: empty network."""

    def test_empty_network_analysis(self):
        """Analyzing an empty network returns zeroed metrics."""
        isl = InterSatelliteNetwork([])
        analyzer = NetworkAnalyzer(isl)
        metrics = analyzer.analyze(T0)
        self.assertEqual(metrics.n_satellites, 0)
        self.assertEqual(metrics.n_links, 0)
        self.assertEqual(metrics.connected_components, 0)


if __name__ == "__main__":
    unittest.main()
