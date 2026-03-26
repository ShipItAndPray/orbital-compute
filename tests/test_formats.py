"""Tests for formats module — TLE, CCSDS OEM, STK, JSON schema support."""

import unittest
from datetime import datetime, timezone

from orbital_compute.orbit import starlink_shell_1_sample
from orbital_compute.formats import (
    export_oem,
    export_stk_ephemeris,
    export_tle,
    generate_synthetic_tle,
    get_simulation_schema,
    parse_tle,
    parse_tle_to_satellites,
    tle_checksum,
    validate_results_against_schema,
    validate_tle,
    validate_tle_checksum,
    wrap_results_with_metadata,
)


T0 = datetime(2026, 3, 26, 0, 0, 0, tzinfo=timezone.utc)


class TestTLEChecksum(unittest.TestCase):
    """Tests for TLE checksum validation."""

    def test_valid_checksum_passes(self):
        """TLE lines from starlink_shell_1_sample have valid checksums."""
        sats = starlink_shell_1_sample(2)
        for sat in sats:
            self.assertTrue(validate_tle_checksum(sat.tle_line1),
                            f"Line 1 checksum failed for {sat.name}")
            self.assertTrue(validate_tle_checksum(sat.tle_line2),
                            f"Line 2 checksum failed for {sat.name}")

    def test_invalid_checksum_fails(self):
        """Corrupted checksum digit is detected."""
        sats = starlink_shell_1_sample(1)
        line1 = sats[0].tle_line1
        # Flip the last digit (checksum)
        original_check = int(line1[68])
        bad_check = (original_check + 1) % 10
        bad_line = line1[:68] + str(bad_check)
        self.assertFalse(validate_tle_checksum(bad_line))

    def test_short_line_fails(self):
        """Line shorter than 69 chars fails validation."""
        self.assertFalse(validate_tle_checksum("1 12345"))

    def test_checksum_computation(self):
        """tle_checksum returns a single digit [0-9]."""
        sats = starlink_shell_1_sample(1)
        cs = tle_checksum(sats[0].tle_line1)
        self.assertIsInstance(cs, int)
        self.assertGreaterEqual(cs, 0)
        self.assertLessEqual(cs, 9)


class TestTLEValidation(unittest.TestCase):
    """Tests for full TLE pair validation."""

    def test_valid_tle_no_errors(self):
        """Valid TLE pair returns no errors."""
        sats = starlink_shell_1_sample(1)
        errors = validate_tle(sats[0].tle_line1, sats[0].tle_line2)
        self.assertEqual(errors, [])

    def test_wrong_line_numbers(self):
        """Lines that don't start with '1 ' and '2 ' produce errors."""
        errors = validate_tle(
            "X" + " " * 68,
            "Y" + " " * 68,
        )
        self.assertGreater(len(errors), 0)

    def test_synthetic_tle_validates(self):
        """Synthetically generated TLE validates correctly."""
        line1, line2 = generate_synthetic_tle(
            "TEST", 99999, 53.0, 120.0, 0.0001, 0.0, 180.0, 15.05,
        )
        errors = validate_tle(line1, line2)
        self.assertEqual(errors, [], f"Synthetic TLE errors: {errors}")


class TestTLEParsing(unittest.TestCase):
    """Tests for TLE parsing and export."""

    def test_roundtrip_3line(self):
        """Export 3-line TLE and re-import gets same satellite count."""
        sats = starlink_shell_1_sample(4)
        text = export_tle(sats, include_names=True)
        parsed = parse_tle(text)
        self.assertEqual(len(parsed), 4)

    def test_roundtrip_2line(self):
        """Export 2-line TLE (no names) and re-import."""
        sats = starlink_shell_1_sample(3)
        text = export_tle(sats, include_names=False)
        parsed = parse_tle(text)
        self.assertEqual(len(parsed), 3)

    def test_parse_to_satellites(self):
        """parse_tle_to_satellites returns Satellite objects."""
        sats = starlink_shell_1_sample(2)
        text = export_tle(sats)
        reimported = parse_tle_to_satellites(text)
        self.assertEqual(len(reimported), 2)
        # Each should produce valid positions
        for sat in reimported:
            pos = sat.position_at(T0)
            self.assertGreater(pos.altitude_km, 400)


class TestCCSDSOEM(unittest.TestCase):
    """Tests for CCSDS OEM export."""

    def test_oem_has_required_headers(self):
        """OEM output contains version, meta start/stop, ref frame."""
        sat = starlink_shell_1_sample(1)[0]
        oem = export_oem(sat, T0, duration_hours=0.5, step_seconds=60.0)
        self.assertIn("CCSDS_OEM_VERS = 2.0", oem)
        self.assertIn("META_START", oem)
        self.assertIn("META_STOP", oem)
        self.assertIn("EME2000", oem)
        self.assertIn("UTC", oem)

    def test_oem_contains_satellite_name(self):
        """OEM contains the satellite name."""
        sat = starlink_shell_1_sample(1)[0]
        oem = export_oem(sat, T0, duration_hours=0.1)
        self.assertIn(sat.name, oem)

    def test_oem_data_point_count(self):
        """OEM has correct number of ephemeris data points."""
        sat = starlink_shell_1_sample(1)[0]
        oem = export_oem(sat, T0, duration_hours=1.0, step_seconds=60.0)
        # 1 hour / 60s = 60 intervals + 1 = 61 points
        data_lines = [l for l in oem.split('\n') if l.strip().startswith('202')]
        self.assertEqual(len(data_lines), 61)


class TestSTKEphemeris(unittest.TestCase):
    """Tests for STK ephemeris export."""

    def test_stk_has_correct_structure(self):
        """STK output has version, BEGIN/END Ephemeris markers."""
        sat = starlink_shell_1_sample(1)[0]
        stk = export_stk_ephemeris(sat, T0, duration_hours=0.5, step_seconds=60.0)
        self.assertIn("stk.v.12.0", stk)
        self.assertIn("BEGIN Ephemeris", stk)
        self.assertIn("END Ephemeris", stk)
        self.assertIn("EphemerisTimePosVel", stk)

    def test_stk_point_count(self):
        """STK has correct NumberOfEphemerisPoints."""
        sat = starlink_shell_1_sample(1)[0]
        stk = export_stk_ephemeris(sat, T0, duration_hours=0.5, step_seconds=60.0)
        expected_points = int(0.5 * 3600 / 60) + 1
        self.assertIn(f"NumberOfEphemerisPoints  {expected_points}", stk)

    def test_stk_coordinate_system(self):
        """STK uses specified coordinate system."""
        sat = starlink_shell_1_sample(1)[0]
        stk = export_stk_ephemeris(sat, T0, duration_hours=0.1, coord_system="ICRF")
        self.assertIn("ICRF", stk)


class TestJSONSchema(unittest.TestCase):
    """Tests for JSON schema and validation."""

    def test_schema_has_required_fields(self):
        """Schema defines all required top-level fields."""
        schema = get_simulation_schema()
        self.assertIn("$schema", schema)
        self.assertIn("properties", schema)
        self.assertIn("satellite_details", schema["properties"])
        self.assertIn("fleet_utilization_pct", schema["properties"])

    def test_valid_result_passes_validation(self):
        """A correct simulation result passes validation."""
        good = {
            "meta": {"version": "1.0.0", "created_at": "2026-03-26T00:00:00",
                     "simulator": "orbital-compute"},
            "config": {"n_satellites": 6, "sim_hours": 6.0, "n_jobs": 20},
            "scheduler": {"total_jobs": 20, "completed": 15, "running": 2, "queued": 3},
            "fleet_utilization_pct": 45.2,
            "total_compute_hours": 8.5,
            "satellite_details": {},
        }
        errors = validate_results_against_schema(good)
        self.assertEqual(errors, [])

    def test_missing_fields_detected(self):
        """Missing required fields are detected."""
        bad = {"fleet_utilization_pct": 50.0}
        errors = validate_results_against_schema(bad)
        self.assertGreater(len(errors), 0)

    def test_out_of_range_utilization_detected(self):
        """Fleet utilization > 100 is flagged."""
        bad = {
            "meta": {}, "config": {}, "scheduler": {},
            "fleet_utilization_pct": 150.0,
            "total_compute_hours": 0,
            "satellite_details": {},
        }
        errors = validate_results_against_schema(bad)
        self.assertTrue(any("out of range" in e for e in errors))

    def test_wrap_metadata_adds_simulator(self):
        """wrap_results_with_metadata adds simulator info."""
        wrapped = wrap_results_with_metadata({"some": "data"})
        self.assertEqual(wrapped["meta"]["simulator"], "orbital-compute")
        self.assertIn("version", wrapped["meta"])
        self.assertIn("created_at", wrapped["meta"])

    def test_completed_jobs_missing_job_id(self):
        """completed_jobs entry without job_id is flagged."""
        result = {
            "meta": {}, "config": {}, "scheduler": {},
            "fleet_utilization_pct": 50.0,
            "total_compute_hours": 1.0,
            "satellite_details": {},
            "completed_jobs": [{"satellite": "SAT-0"}],  # missing job_id
        }
        errors = validate_results_against_schema(result)
        self.assertTrue(any("job_id" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
