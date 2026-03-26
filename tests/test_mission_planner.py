"""Tests for mission_planner module."""
import unittest
from datetime import datetime, timezone, timedelta
from orbital_compute.mission_planner import MissionPlanner, TimedEvent
from orbital_compute.orbit import starlink_shell_1_sample
from orbital_compute.ground_stations import DEFAULT_GROUND_STATIONS


class TestMissionPlanner(unittest.TestCase):
    def setUp(self):
        self.sats = starlink_shell_1_sample(4)
        self.start = datetime(2026, 3, 26, 0, 0, 0, tzinfo=timezone.utc)
        self.stations = DEFAULT_GROUND_STATIONS[:3]

    def test_timeline_not_empty(self):
        planner = MissionPlanner(self.sats, self.stations)
        timeline = planner.generate_timeline(self.start, duration_hours=2.0)
        self.assertGreater(len(timeline), 0)

    def test_timeline_events_have_required_fields(self):
        planner = MissionPlanner(self.sats, self.stations)
        timeline = planner.generate_timeline(self.start, duration_hours=1.0)
        for event in timeline[:10]:
            self.assertIsInstance(event.satellite, str)
            self.assertIsInstance(event.event_type, str)
            self.assertIsInstance(event.timestamp, datetime)

    def test_timeline_sorted_by_time(self):
        planner = MissionPlanner(self.sats, self.stations)
        timeline = planner.generate_timeline(self.start, duration_hours=2.0)
        for i in range(len(timeline) - 1):
            self.assertLessEqual(timeline[i].timestamp, timeline[i + 1].timestamp)

    def test_eclipse_events_exist(self):
        planner = MissionPlanner(self.sats, self.stations)
        timeline = planner.generate_timeline(self.start, duration_hours=3.0)
        eclipse_events = [e for e in timeline if 'eclipse' in e.event_type]
        self.assertGreater(len(eclipse_events), 0)

    def test_pass_prediction(self):
        planner = MissionPlanner(self.sats, self.stations)
        passes = planner.predict_passes(self.start, duration_hours=6.0)
        # With 4 sats and 3 stations over 6h, should find some passes
        self.assertGreater(len(passes), 0)


if __name__ == '__main__':
    unittest.main()
