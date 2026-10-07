"""Synthetic loss-threshold and re-detection reason checks; no models or ROS."""
import contextlib
import unittest
from unittest.mock import Mock, patch
import numpy as np
from cloud_track.pipeline.cloud_track import CloudTrack
from cloud_track.foundation_model_wrappers import aware_candidates as candidates


class Item10Tests(unittest.TestCase):
    def target(self):
        target = CloudTrack.__new__(CloudTrack)
        target.box = np.array([10, 10, 30, 70], dtype=np.float32)
        target._anchor_box = target.box.copy()
        target._anchor_ground = None
        target._anchor_time = 0.0
        target._redetect_misses = 0
        target.max_redetect_misses = 2
        target.lost_timeout_s = 30.0
        target.fm_timer = contextlib.nullcontext()
        target.backend = Mock()
        target.backend.redetect_target.return_value = (None, 'missing: 2 people in view; 1 near last position with wrong colors')
        return target

    def check(self, target, now):
        return target._redetect(np.zeros((100, 100, 3), dtype=np.uint8), 'person', 'red top', None, now)

    def test_last_box_kept_until_both_loss_conditions(self):
        target = self.target()
        for now in (3.0, 6.0, 29.9):
            box, success = self.check(target, now)
            self.assertTrue(success)
            np.testing.assert_array_equal(box, target.box)
        self.assertEqual(self.check(target, 30.0), (None, False))
        target = self.target()
        self.assertTrue(self.check(target, 31.0)[1])  # Time alone is not sufficient.
        self.assertEqual(self.check(target, 32.0), (None, False))

    def test_miss_reason_counts_and_reaches_warning(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        boxes = np.array([[10, 10, 30, 70], [50, 10, 70, 70]])
        with patch.object(candidates, 'plausible', return_value=True), patch.object(candidates, 'color_score', return_value=(0.0, 0.0)):
            box, reason = candidates.verify_target(image, boxes, None, None, boxes[0], 'red top',
                anchor_distance=lambda b: 1.0 if b[0] == 10 else None)
        self.assertIsNone(box)
        self.assertIn('2 people in view', reason)
        self.assertIn('1 near last position with wrong colors', reason)
        target = self.target()
        target.backend.redetect_target.return_value = (None, reason)
        with patch('cloud_track.pipeline.cloud_track.logger.warning') as warning:
            self.check(target, 3)
        self.assertIn(reason, warning.call_args.args[0])
        self.assertIn('0 people in view', candidates.verify_target(image, [], None, None, None, 'red top')[1])


if __name__ == '__main__':
    unittest.main()
