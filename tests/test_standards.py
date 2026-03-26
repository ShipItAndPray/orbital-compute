"""Tests for ECSS standards compliance module."""
import unittest
from orbital_compute.standards import (
    PowerBudget, ThermalBudget, ThermalBudgetItem,
    slant_range_km, calculate_link_budget,
)


class TestPowerBudget(unittest.TestCase):
    def test_add_items_and_total(self):
        pb = PowerBudget(satellite_name="TEST-SAT", solar_panel_eol_watts=2000)
        pb.add_item("OBC", "nominal", 50)
        pb.add_item("GPU", "nominal", 500)
        self.assertAlmostEqual(pb.total_power_w("nominal"), 550, delta=1)

    def test_margins_increase_total(self):
        pb = PowerBudget(satellite_name="TEST", solar_panel_eol_watts=2000)
        pb.add_item("GPU", "nominal", 500)
        self.assertGreater(pb.total_with_margins_w("nominal"), 500)


class TestSlantRange(unittest.TestCase):
    def test_zenith_equals_altitude(self):
        r = slant_range_km(550, 90)
        self.assertAlmostEqual(r, 550, delta=5)

    def test_low_elevation_longer(self):
        self.assertGreater(slant_range_km(550, 10), slant_range_km(550, 45))

    def test_positive(self):
        self.assertGreater(slant_range_km(550, 5), 0)


class TestLinkBudget(unittest.TestCase):
    def test_x_band_closes(self):
        result = calculate_link_budget(band="x_band", altitude_km=550,
                                        elevation_deg=10, data_rate_bps=100e6)
        self.assertGreater(result.link_margin_db, 0)

    def test_higher_altitude_lower_margin(self):
        r_low = calculate_link_budget(band="x_band", altitude_km=550, elevation_deg=30)
        r_high = calculate_link_budget(band="x_band", altitude_km=1500, elevation_deg=30)
        self.assertGreater(r_low.link_margin_db, r_high.link_margin_db)


if __name__ == "__main__":
    unittest.main()
