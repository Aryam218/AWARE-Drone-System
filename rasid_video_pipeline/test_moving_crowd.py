import unittest
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
from PIL import Image
from analytics.moving_crowd import MovingCrowdCounter
from analytics.shared_detector import shared_detector
from cloud_track.foundation_model_wrappers import aware_geo as geo

class Detector:
    box_threshold=0.1
    text_threshold=0.05
    def __init__(self): self.calls=0;self.active=0;self.max_active=0
    def run_inference(self,image,**kwargs):
        self.calls+=1;self.active+=1;self.max_active=max(self.max_active,self.active)
        time.sleep(.01);self.active-=1
        return image,None,np.array([[490,350,510,400]],dtype=float),np.array([.9])

class MovingCrowdTests(unittest.TestCase):
    def setUp(self):
        self.raw=Detector();self.counter=MovingCrowdCounter(detector=self.raw)
        lat,lon=self.counter.world_to_geo(0,5)
        self.pose=dict(lat=lat,lon=lon,alt_rel_m=15,heading_deg=0,roll_deg=0,pitch_deg=0,fx=400,fy=400,cx=500,cy=400)

    def test_inverse_attitude(self):
        for roll,pitch,heading in [(0,0,0),(12,-8,124),(-5,13,275)]:
            pose=dict(self.pose,roll_deg=roll,pitch_deg=pitch,heading_deg=heading)
            for u,v in [(500,400),(330,650),(700,200)]:
                point=geo.pixel_to_ground(u,v,pose)
                pixel=geo.ground_to_pixel(*point,pose)
                np.testing.assert_allclose(pixel[:2],[u,v],atol=1e-6)

    def test_boxes_not_tracks_and_filters(self):
        boxes=[[490,350,510,400],[491,350,511,400],[0,0,1000,800],[0,0,180,600]]
        first=self.counter.count_detections(boxes,[.9,.8,.7,.6],self.pose,1000,800,1)
        self.assertEqual(first['total_people'],1)
        shifted=self.counter.count_detections([[700,400,725,475]],[.9],self.pose,1000,800,13)
        self.assertEqual(shifted['total_people'],1)
        self.assertNotIn('footfall',first);self.assertNotIn('dwell',first)

    def test_zone_world_roundtrip_and_feet(self):
        point=self.counter.world_to_geo(-25,-12)
        np.testing.assert_allclose(self.counter.geo_to_world(*point),[-25,-12],atol=1e-8)
        self.assertEqual(self.counter.zone_at(*point),'booth_a_front')
        point=self.counter.world_to_geo(0,12)
        u,v,_=geo.ground_to_pixel(*point,self.pose)
        result=self.counter.count_detections([[u-10,v-40,u+10,v]],[.9],self.pose,1000,800,7)
        self.assertEqual(result['people'][0]['zone'],'plaza')
        self.assertEqual(next(z for z in result['zones'] if z['id']=='plaza')['visible_count'],1)

    def test_coverage_occupancy_last_seen(self):
        self.counter.coverage=lambda *args:{z['id']:.8 for z in self.counter.zones}
        result=self.counter.count_detections([],[],self.pose,1000,800,10)
        self.assertTrue(all(z['occupancy_percent']==0 and z['last_seen']==10 for z in result['zones']))
        self.counter.coverage=lambda *args:{z['id']:.799 for z in self.counter.zones}
        result=self.counter.count_detections([],[],self.pose,1000,800,11)
        self.assertTrue(all(z['occupancy_percent'] is None for z in result['zones']))
        self.counter.coverage=lambda *args:{z['id']:0 for z in self.counter.zones}
        result=self.counter.count_detections([],[],self.pose,1000,800,12)
        self.assertTrue(all(z['last_seen']==11 for z in result['zones']))
        result=self.counter.count_detections([],[],None,1000,800,13)
        self.assertTrue(all(z['visible_count'] is None and z['coverage_fraction'] is None for z in result['zones']))

    def test_actual_coverage(self):
        coverage=self.counter.coverage(self.pose,1000,800)
        self.assertTrue(all(0<=x<=1 for x in coverage.values()))
        self.assertGreater(coverage['plaza'],.5)
        self.assertEqual(coverage['entrance'],0)

    def test_runner_acknowledges_before_optional_crowd_inference(self):
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        import pipeline_runner as runner
        frame=np.zeros((800,1000,3),dtype=np.uint8)
        order=[]
        original=self.raw.run_inference
        def detect(*args,**kwargs):
            order.append('inference')
            return original(*args,**kwargs)
        self.raw.run_inference=detect
        event=runner.ConfirmedEvent(frame,np.array([490,350,510,400]),1,0,1)
        backend=SimpleNamespace(detector=self.counter.detector,frame_pose=self.pose)
        controller=SimpleNamespace(pipeline=SimpleNamespace(backend=backend),process_frame=Mock(return_value=event))
        with patch.object(runner,'SearchController',return_value=controller), patch.object(runner,'open_stream',return_value=[frame]), patch.object(runner,'VideoStreamer',return_value=[frame]):
            runner.run_on_video('synthetic','person','red top',lambda e:None,lambda e:None,lambda e:None,
                                on_confirmed=lambda e:order.append('confirm'),on_crowd_analysis=lambda r:order.append('crowd'))
        self.assertEqual(order,['confirm','inference','crowd'])
        self.assertEqual(self.raw.calls,1)

    def test_search_backend_and_crowd_share_raw_detections(self):
        from unittest.mock import Mock
        from cloud_track.foundation_model_wrappers.detector_vlm_pipeline import DetectorVlmPipeline
        backend=DetectorVlmPipeline(vlm=Mock(),detector=self.raw)
        backend.person_tracker.update=Mock(return_value=[])
        backend.run_inference_inner('person','red top',Image.new('RGB',(1000,800)))
        counter=MovingCrowdCounter(detector=backend.detector)
        result=counter.analyze_frame(np.zeros((800,1000,3),dtype=np.uint8),self.pose,2)
        self.assertIs(counter.detector,backend.detector)
        self.assertEqual(result['total_people'],1)
        self.assertEqual(self.raw.calls,1)
        self.assertEqual(counter.detector.cache_hits,1)

    def test_same_frame_reuse_mutation_and_serialization(self):
        shared=shared_detector(self.raw)
        self.assertIs(shared,self.counter.detector)
        image=Image.new('RGB',(1000,800))
        shared.run_inference(image,prompt='person',mark_results=False)[2][:]=0
        result=self.counter.analyze_frame(np.zeros((800,1000,3),dtype=np.uint8),self.pose,1)
        self.assertEqual(result['total_people'],1);self.assertEqual(self.raw.calls,1)
        with ThreadPoolExecutor(2) as pool:
            futures=[pool.submit(shared.run_inference,Image.new('RGB',(10,10),color=c),prompt='person') for c in ('red','blue')]
            for f in futures:f.result()
        self.assertEqual(self.raw.max_active,1)
        self.assertEqual(self.raw.calls,3)

if __name__=='__main__':unittest.main()
