"""Confirm refresh and low-detail GPT checks without models, API or ROS."""
import contextlib
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
from PIL import Image
import pipeline_runner as runner
from cloud_track.pipeline.cloud_track import CloudTrack
from cloud_track.foundation_model_wrappers import detector_vlm_pipeline as detector
from cloud_track.foundation_model_wrappers.gpt_four_wrapper import GPTFourWrapper

class Followups(unittest.TestCase):
    def test_confirm_ack_precedes_forced_next_frame_check(self):
        target=CloudTrack(Mock(),Mock())
        target.tracker_initialized=True
        target.current_track_id=7
        target._last_redetect_time=10.0
        controller=runner.SearchController.__new__(runner.SearchController)
        controller.pipeline=target
        controller.category='person';controller.description='red top'
        controller._parent_confirmed_pending=True;controller._parent_rejected_pending=False
        controller._late_reject_pending=False;controller._candidate_confirmed=False
        controller._last_candidate_frame=np.zeros((100,100,3),dtype=np.uint8)
        controller._last_candidate_bbox=np.array([10,10,30,70])
        with patch.object(target,'forward',return_value=(controller._last_candidate_bbox,None,.8)) as forward:
            acknowledgement=controller.process_frame(controller._last_candidate_frame,1,timestamp=10.1)
            self.assertIsInstance(acknowledgement,runner.ConfirmedEvent)
            forward.assert_not_called()  # The acknowledgement does not wait for detector work.
            self.assertTrue(target._redetect_due(10.2)) # Only 0.2 s since last check, below 3 s.
            controller.process_frame(controller._last_candidate_frame,2,timestamp=10.2)
            forward.assert_called_once()
        target.fm_timer=contextlib.nullcontext()
        target.box=controller._last_candidate_bbox.copy();target._anchor_box=target.box.copy()
        target._anchor_time=10.0
        target.backend.redetect_target.return_value=(None,'missing: 0 people in view')
        target._redetect(controller._last_candidate_frame,'person','red top',target.box,10.2)
        self.assertFalse(target._force_redetect)
        self.assertFalse(target._redetect_due(10.3)) # The forced request is consumed once.
        self.assertTrue(target._redetect_due(13.2))
        target.request_redetection();target.reset();self.assertFalse(target._force_redetect)

    def test_forced_detection_runs_before_interval_then_resumes_schedule(self):
        backend=Mock();backend.frame_pose=None
        backend.ground_position.return_value=None
        target=CloudTrack(backend,Mock())
        target.tracker_initialized=True
        target.box=np.array([10,10,30,70]);target._anchor_box=target.box.copy()
        target._anchor_time=10.0;target._last_redetect_time=10.0
        target.track_frame=Mock(return_value=(target.box,True,.9))
        backend.redetect_target.return_value=(target.box,'same')
        frame=np.zeros((100,100,3),dtype=np.uint8)
        target.request_redetection()
        target.forward(frame,'person','red top',timestamp=10.1)
        backend.redetect_target.assert_called_once()
        self.assertEqual(target._anchor_time,10.1)
        target.forward(frame,'person','red top',timestamp=10.2)
        backend.redetect_target.assert_called_once()
        target.forward(frame,'person','red top',timestamp=13.2)
        self.assertEqual(backend.redetect_target.call_count,2)

    def test_get_vlm_uses_low_detail_and_payload_preserves_it(self):
        with patch.object(detector,'GPTFourWrapper') as wrapper:
            detector.get_vlm('gpt-4o-mini','red top',0)
            self.assertEqual(wrapper.call_args.kwargs['image_detail'],'low')
        obj=GPTFourWrapper.__new__(GPTFourWrapper)
        obj.image_detail='low';obj.system_prompt=None;obj.model='gpt-4o-mini'
        obj.encode_image=Mock(return_value='jpeg')
        payload=obj._build_payload('person',Image.new('RGB',(10,10)))
        self.assertEqual(payload['messages'][-1]['content'][-1]['image_url']['detail'],'low')

if __name__=='__main__':unittest.main()
