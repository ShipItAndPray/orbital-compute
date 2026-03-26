"""Tests for 3D visualization generator."""
import os
import tempfile
import unittest
from orbital_compute.visualize import generate_3d_visualization
from orbital_compute.orbit import starlink_shell_1_sample
from orbital_compute.ground_stations import DEFAULT_GROUND_STATIONS


class TestVisualization(unittest.TestCase):
    def test_generates_html_file(self):
        sats = starlink_shell_1_sample(3)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            result = generate_3d_visualization(sats, DEFAULT_GROUND_STATIONS[:2],
                                                hours=0.5, output=path, open_browser=False)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 1000)
        finally:
            os.unlink(path)

    def test_html_contains_threejs(self):
        sats = starlink_shell_1_sample(2)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            generate_3d_visualization(sats, hours=0.5, output=path, open_browser=False)
            with open(path) as f:
                html = f.read()
            self.assertIn("three", html.lower())
            self.assertIn("satellite", html.lower())
        finally:
            os.unlink(path)

    def test_returns_path(self):
        sats = starlink_shell_1_sample(2)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            result = generate_3d_visualization(sats, hours=0.5, output=path, open_browser=False)
            self.assertEqual(result, path)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
