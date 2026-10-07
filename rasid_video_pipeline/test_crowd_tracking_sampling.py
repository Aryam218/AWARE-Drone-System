"""Explain sparse-view count behavior without loading a neural model."""
import unittest
from unittest.mock import Mock,patch
import numpy as np
from analytics.crowd_analytics import CrowdAnalytics
class SparseTrackingTests(unittest.TestCase):
    def test_new_view_detections_can_count_zero_until_confirmed(self):
        first=np.array([[10,10,50,100],[90,10,130,100]],dtype=np.float32)
        new=np.array([[450,300,490,400],[550,300,590,400]],dtype=np.float32)
        detector=Mock();detector.run_inference.side_effect=[(None,None,b,[.9,.9]) for b in [first,new,new]]
        with patch('analytics.crowd_analytics.GroundingDinoHuggingfaceWrapper',return_value=detector):
            crowd=CrowdAnalytics()
        image=np.zeros((480,640,3),dtype=np.uint8)
        counts=[crowd.analyze_frame(image,timestamp=t)[1] for t in [0,12,24]]
        self.assertEqual(counts,[2,0,2])
        print('Two detections in each frame; analytics counts after view change:',counts)
if __name__=='__main__':unittest.main()
