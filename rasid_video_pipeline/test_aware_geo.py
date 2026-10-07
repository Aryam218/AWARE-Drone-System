"""Synthetic body-fixed camera projection checks; no ROS or detector needed."""
import math
import unittest
from unittest.mock import patch
from cloud_track.foundation_model_wrappers import aware_geo as geo


class PixelToGroundTest(unittest.TestCase):
    def setUp(self):
        self.pose = dict(lat=0.0, lon=0.0, alt_rel_m=10.0, heading_deg=0.0,
                         fx=500.0, fy=500.0, cx=320.0, cy=240.0)

    def ground_m(self, **attitude):
        point = geo.pixel_to_ground(320, 240, dict(self.pose, **attitude))
        return None if point is None else tuple(v * geo.METERS_PER_DEG_LAT for v in point)

    def test_level_center(self):
        north, east = self.ground_m()
        self.assertAlmostEqual(north, 10.0)
        self.assertAlmostEqual(east, 0.0)
        self.assertEqual(self.ground_m(), self.ground_m(roll_deg=None, pitch_deg=None))
        print(f"Level at 10 m: {north:.3f} m ahead")

    def test_nose_down_center(self):
        north, east = self.ground_m(pitch_deg=-8.0)
        self.assertAlmostEqual(north, 10.0 / math.tan(math.radians(53)))
        self.assertAlmostEqual(north, 7.5, delta=0.05)
        self.assertAlmostEqual(east, 0.0)
        print(f"Nose down 8 degrees: {north:.3f} m ahead")

    def test_heading_and_roll(self):
        north, east = self.ground_m(heading_deg=90.0)
        self.assertAlmostEqual(north, 0.0)
        self.assertAlmostEqual(east, 10.0)
        north, east = self.ground_m(roll_deg=10.0)
        self.assertAlmostEqual(north, 10.0 / math.cos(math.radians(10)))
        self.assertAlmostEqual(east, -10.0 * math.tan(math.radians(10)))

    def test_stabilized_and_horizon(self):
        with patch.object(geo, "CAMERA_STABILIZED", True):
            self.assertEqual(self.ground_m(roll_deg=20, pitch_deg=-8), self.ground_m())
        self.assertIsNone(self.ground_m(pitch_deg=50.0))


if __name__ == "__main__":
    unittest.main()
