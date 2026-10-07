"""Quota regression tests: mock HTTP, models and video; no API calls or windows."""
from datetime import timedelta
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
from PIL import Image
from cloud_track.foundation_model_wrappers import gpt_four_wrapper as gpt
from cloud_track.foundation_model_wrappers import detector_vlm_pipeline as detection
import pipeline_runner as runner
import run_live as live


def response(status, error=None):
    return SimpleNamespace(status_code=status, text=str(error or ''), headers={},
        json=lambda: {'error': error} if error else {'choices': [{'message': {'content': 'Decision: MATCH'}}]},
        elapsed=timedelta(seconds=0.1))


class QuotaTests(unittest.TestCase):
    def wrapper(self):
        # Avoid credentials, cache files and model construction.
        obj = gpt.GPTFourWrapper.__new__(gpt.GPTFourWrapper)
        obj.max_retries = 1
        obj.max_retry_wait_s = 2.0
        obj.headers = {}
        obj.connect_timeout_s = obj.read_timeout_s = 1
        return obj

    def test_insufficient_quota_never_retries(self):
        for field in ('type', 'code'):
            with self.subTest(field=field), patch.object(gpt.requests, 'post', return_value=response(429, {field: 'insufficient_quota'})) as post, patch.object(gpt.time, 'sleep') as sleep:
                with self.assertRaises(gpt.VlmInsufficientQuotaError):
                    self.wrapper()._post_with_retries({})
                self.assertEqual(post.call_count, 1)
                sleep.assert_not_called()

    def test_temporary_rate_limit_still_retries(self):
        with patch.object(gpt.requests, 'post', side_effect=[response(429, {'type': 'rate_limit_exceeded'}), response(200)]) as post, patch.object(gpt.time, 'sleep') as sleep:
            result, _ = self.wrapper()._post_with_retries({})
            self.assertEqual(result, 'Decision: MATCH')
            self.assertEqual(post.call_count, 2)
            sleep.assert_called_once()

    def test_pipeline_disables_calls_and_logs_error_once(self):
        image = Image.new('RGB', (100, 100))
        detector = Mock()
        detector.run_inference.return_value = (image, None, np.array([[10, 10, 30, 80]]), [0.9])
        vlm = Mock()
        vlm.run_inference.side_effect = gpt.VlmInsufficientQuotaError(gpt.NO_CREDITS_MESSAGE)
        with patch.object(detection, 'ByteTrackWrapper'), patch.object(detection, 'select_candidates', return_value=[{'bbox': [10, 10, 30, 80], 'track_id': 1}]), patch.object(detection.logger, 'error') as error:
            pipeline = detection.DetectorVlmPipeline(vlm, detector, enable_overscan=False)
            try:
                for _ in range(2):
                    responses = pipeline.run_inference_inner('person', 'red top', image)[0]
                    self.assertEqual(responses, [detection.RESPONSE_VERIFICATION_OFF])
                self.assertEqual(pipeline.verification_error, gpt.NO_CREDITS_MESSAGE)
                self.assertEqual(vlm.run_inference.call_count, 1)
                error.assert_called_once_with(gpt.NO_CREDITS_MESSAGE)
                self.assertEqual(pipeline.vlm_cache, {})
            finally:
                pipeline.shutdown()

    def test_failure_reaches_live_status_bar_once_and_persists(self):
        frame = np.zeros((100, 960, 3), dtype=np.uint8)
        controller = Mock()
        controller.pipeline.backend.verification_error = gpt.NO_CREDITS_MESSAGE
        controller.process_frame.return_value = None
        notifications = []
        with patch.object(runner, 'SearchController', return_value=controller), patch.object(runner, 'open_stream', return_value=[]), patch.object(runner, 'VideoStreamer', return_value=[frame, frame]):
            runner.run_on_video('mock.mp4', 'person', 'red top', Mock(), Mock(), Mock(),
                                on_verification_error=notifications.append)
        self.assertEqual(len(notifications), 1)
        self.assertEqual(notifications[0].code, 'insufficient_quota')
        with patch.dict(live._overlay, dict(state='Searching', time=0, bbox=None), clear=True):
            live._on_verification_error(notifications[0])
            live._on_ai_frame(None, None)
            with patch.object(live.cv2, 'putText') as put_text:
                live._draw(frame)
            self.assertEqual(put_text.call_args_list[0].args[1], gpt.NO_CREDITS_MESSAGE)
            self.assertEqual(put_text.call_args_list[0].args[5], (0, 0, 255))


if __name__ == '__main__':
    unittest.main()
