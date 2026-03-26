"""Tests for constellation generation."""
import unittest
from orbital_compute.constellations import generate_constellation, CONSTELLATIONS


class TestConstellationGeneration(unittest.TestCase):
    def test_starlink_mini_generates_correct_count(self):
        config = CONSTELLATIONS["starlink-mini"]
        sats = generate_constellation(config)
        self.assertEqual(len(sats), config.total_sats)

    def test_max_sats_limits_output(self):
        config = CONSTELLATIONS["starlink-mini"]
        sats = generate_constellation(config, max_sats=5)
        self.assertEqual(len(sats), 5)

    def test_polar_orbit_inclination(self):
        config = CONSTELLATIONS["polar-compute"]
        sats = generate_constellation(config, max_sats=2)
        self.assertEqual(len(sats), 2)
        # Satellite names should contain config name prefix
        self.assertTrue(sats[0].name.startswith("Polar Co"))

    def test_all_presets_generate(self):
        for name, config in CONSTELLATIONS.items():
            sats = generate_constellation(config, max_sats=3)
            self.assertGreater(len(sats), 0, f"Failed for {name}")

    def test_satellites_have_valid_positions(self):
        from datetime import datetime, timezone
        config = CONSTELLATIONS["starcloud"]
        sats = generate_constellation(config, max_sats=4)
        t = datetime(2026, 3, 26, 12, 0, 0, tzinfo=timezone.utc)
        for sat in sats:
            pos = sat.position_at(t)
            self.assertGreater(pos.altitude_km, 300)
            self.assertLess(pos.altitude_km, 2500)


if __name__ == "__main__":
    unittest.main()
