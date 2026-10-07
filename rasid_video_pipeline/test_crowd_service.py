import asyncio
import tempfile
from pathlib import Path
import yaml
from analytics.moving_crowd import MovingCrowdCounter
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
from PIL import Image
from analytics.crowd_service import CrowdService, protocol_analytics
from analytics.shared_detector import shared_detector
from test_moving_crowd import Detector

class CrowdServiceTests(unittest.TestCase):
    def setUp(self):
        self.raw=Detector();self.publish=Mock();self.service=CrowdService(self.raw,self.publish)
        self.frame=np.zeros((800,1000,3),dtype=np.uint8)
        self.pose=dict(lat=24.7136,lon=46.6753,alt_rel_m=15,heading_deg=0,fx=400,fy=400,cx=500,cy=400)
        self.pipeline=SimpleNamespace(tracker_initialized=True,_force_redetect=False,_last_redetect_time=10,redetect_interval_s=3)
        self.controller=SimpleNamespace(pipeline=self.pipeline)

    def test_zone_metadata_is_read_once_from_custom_yaml(self):
        config = dict(venue=dict(location=dict(latitude=24.7136, longitude=46.6753)),
                      zones=[dict(id='custom_front', type='booth_front', display_name='Custom booth front',
                                  booth='custom_booth', capacity=37, polygon=[[0,0],[1,0],[1,1],[0,1]])])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'zones.yaml'
            path.write_text(yaml.safe_dump(config))
            counter = MovingCrowdCounter(detector=self.raw, zones_path=path)
            # Changing the file after startup must not change the loaded metadata.
            config['zones'][0].update(display_name='Changed', capacity=99)
            path.write_text(yaml.safe_dump(config))
            result = counter.count_detections([], [], None, 1000, 800, 10)
            zones = protocol_analytics(result)['zones']
            self.assertEqual(set(zones), {'custom_front'})
            zone = zones['custom_front']
            self.assertEqual({k: zone[k] for k in ('id','type','display_name','booth','capacity')},
                             dict(id='custom_front', type='booth_front', display_name='Custom booth front',
                                  booth='custom_booth', capacity=37))
            newer = MovingCrowdCounter(detector=self.raw, zones_path=path)
            self.assertEqual(newer.zones[0]['display_name'], 'Changed')
            self.assertEqual(newer.zones[0]['capacity'], 99)

    def test_idle_cadence_pause_and_resume(self):
        self.assertTrue(self.service.process_idle_frame(self.frame,self.pose,1,wall_now=0))
        self.assertFalse(self.service.process_idle_frame(self.frame,self.pose,2,wall_now=9))
        self.assertTrue(self.service.process_idle_frame(self.frame,self.pose,3,wall_now=10))
        self.service.set_search_active(True)
        self.assertFalse(self.service.process_idle_frame(self.frame,self.pose,4,wall_now=25))
        self.service.set_search_active(False)
        self.assertTrue(self.service.process_idle_frame(self.frame,self.pose,5,wall_now=26))
        image=self.publish.call_args.args[1]
        self.assertEqual(image.shape[1],640)

    def test_search_reuse_needs_no_inference_and_current_stamp(self):
        self.service.detector.run_inference(Image.new('RGB',(1000,800)),prompt='person')
        self.service.set_search_active(True)
        self.service.on_search_frame(self.frame,self.pose,20,self.controller)
        self.assertEqual(self.raw.calls,1)
        self.assertEqual(self.publish.call_args.args[0]['sim_time'],20)
        self.assertIn('total_people_in_view',self.publish.call_args.args[0])

    def test_tracking_extra_throttled_and_budgeted(self):
        self.service.detector.inference_seconds=0.1
        self.pipeline.redetect_interval_s=60
        with patch('analytics.crowd_service.time.monotonic',return_value=20):
            self.service.on_search_frame(self.frame,self.pose,11,self.controller)
        self.assertEqual(self.raw.calls,1)
        other=self.frame.copy();other[0,0]=255
        with patch('analytics.crowd_service.time.monotonic',return_value=34):
            self.service.on_search_frame(other,self.pose,12,self.controller)
        self.assertEqual(self.raw.calls,1)
        with patch('analytics.crowd_service.time.monotonic',return_value=35):
            self.service.on_search_frame(other,self.pose,13,self.controller)
        self.assertEqual(self.raw.calls,2)

    def test_due_forced_and_cpu_slow_checks_win(self):
        self.service.on_search_frame(self.frame,self.pose,10.1,self.controller)
        self.assertEqual(self.raw.calls,0)
        self.service.detector.inference_seconds=.1
        self.pipeline._force_redetect=True
        self.service.on_search_frame(self.frame,self.pose,10.1,self.controller)
        self.assertEqual(self.raw.calls,0)
        self.pipeline._force_redetect=False
        self.service.on_search_frame(self.frame,self.pose,13,self.controller)
        self.assertEqual(self.raw.calls,0)
        self.pipeline.tracker_initialized=False
        self.service.on_search_frame(self.frame,self.pose,14,self.controller)
        self.assertEqual(self.raw.calls,0)

    def test_background_rechecks_search_and_never_queues(self):
        image=Image.new('RGB',(10,10))
        self.assertIsNone(self.service.detector.try_background_inference(image,Mock(side_effect=[True,False])))
        self.assertEqual(self.raw.calls,0)
        entered=threading.Event();release=threading.Event()
        def hold():
            with self.service.detector._lock:entered.set();release.wait(2)
        t=threading.Thread(target=hold);t.start();entered.wait(1)
        try:self.assertIsNone(self.service.detector.try_background_inference(image,lambda:True))
        finally:release.set();t.join()
        self.assertEqual(self.raw.calls,0)

    def test_protocol_unknown_partial_and_full(self):
        zones=[dict(id='partial',type='area',display_name='partial label',booth=None,visible_count=8,capacity=10,coverage_fraction=.79,last_seen=5,occupancy_percent=80),
               dict(id='full',type='area',display_name='full label',booth=None,visible_count=9,capacity=10,coverage_fraction=.8,last_seen=5,occupancy_percent=90),
               dict(id='unseen',type='area',display_name='unseen label',booth=None,visible_count=0,capacity=10,coverage_fraction=0,last_seen=3,occupancy_percent=None)]
        result=protocol_analytics(dict(timestamp=5,total_people=20,zones=zones))
        self.assertIsNone(result['zones']['partial']['occupancy_percent'])
        self.assertIsNone(result['zones']['partial']['crowded'])
        self.assertTrue(result['zones']['full']['crowded'])
        self.assertIsNone(result['zones']['unseen']['count'])
        self.assertEqual(result['zones']['unseen']['last_seen_sim_time'],3)
        self.assertNotIn('footfall',result)

    def test_factory_injection_never_loads_second_detector(self):
        from cloud_track.foundation_model_wrappers import detector_vlm_pipeline as module
        with patch.object(module,'get_detector',side_effect=AssertionError('second load')),patch.object(module,'get_vlm',return_value=Mock()):
            search=module.get_vlm_pipeline('gpt-4o-mini','red',False,'sam_lq',detector=self.service.detector)
        self.assertIs(search.detector,self.service.detector)
        import pipeline_runner as runner
        with patch.object(runner,'get_vlm_pipeline',return_value=search) as factory,patch.object(runner,'OpenCVWrapper'):
            runner.SearchController('person','red',detector=self.service.detector)
        self.assertIs(factory.call_args.kwargs['detector'],self.service.detector)

    def test_backend_startup_one_model_and_shutdown(self):
        import backend
        with patch.object(backend,'create_crowd_service',return_value=self.service) as factory,patch.object(self.service,'start'),patch.object(self.service,'close'),patch.object(backend,'_run_thread',None):
            async def lifecycle():
                await backend._capture_loop()
                self.assertIs(backend._crowd.detector,self.service.detector)
                await backend._stop_workers()
            asyncio.run(lifecycle())
        factory.assert_called_once()

class CrowdSocketTests(unittest.TestCase):
    def setUp(self):
        import backend as b
        self.b=b
        b._crowd_latest=b._snapshot_latest=None
        b._status=dict(type='search_status',state='idle',sim_time=None,message='Ready',request_id=None)
        b._candidate=b._location=b._quota=None

    def test_real_socket_crowd_messages_and_replay(self):
        from fastapi.testclient import TestClient
        from tools.auto_dashboard_ws_test import validate
        frame=np.zeros((480,640,3),dtype=np.uint8)
        counter=CrowdService(Detector(),Mock()).counter
        pose=dict(lat=24.7136,lon=46.6753,alt_rel_m=15,heading_deg=0,fx=300,fy=300,cx=320,cy=240)
        message=protocol_analytics(counter.count_detections([],[],pose,640,480,10))
        with patch.object(self.b,'create_crowd_service',return_value=None),TestClient(self.b.app) as client:
            with client.websocket_connect('/ws') as ws:
                ws.receive_json()
                self.b.publish_crowd(message,frame)
                analytics=ws.receive_json();snapshot=ws.receive_json()
                validate(analytics);validate(snapshot)
                self.assertEqual(snapshot['sim_time'],analytics['sim_time'])
                self.assertEqual(analytics['total_people_in_view'],0)
            with client.websocket_connect('/ws') as ws:
                self.assertEqual(ws.receive_json()['state'],'idle')
                self.assertEqual(ws.receive_json(),analytics)
                self.assertEqual(ws.receive_json(),snapshot)

    def test_crowd_failure_isolated_search_injection_and_resume(self):
        from fastapi.testclient import TestClient
        from pipeline_runner import CandidateEvent,ConfirmedEvent
        frame=np.zeros((64,64,3),dtype=np.uint8);box=np.array([5,5,30,50])
        service=Mock();service.on_search_frame.side_effect=ValueError('bad crowd pose')
        done=threading.Event()
        def run(**kw):
            self.assertIs(kw['detector'],service.detector)
            kw['on_target_position'](dict(location={'lat':24,'lon':46},drone={'lat':25,'lon':47},position_sim_time=1))
            kw['on_candidate'](CandidateEvent(frame,box,'red top',1,1,1))
            kw['on_processed_frame'](frame,None,1,Mock())
            while not kw['should_confirm']():time.sleep(.005)
            kw['on_confirmed'](ConfirmedEvent(frame,box,1,2,2))
            done.wait(2)
        with patch.object(self.b,'create_crowd_service',return_value=service),patch.object(self.b,'run_on_video',run),patch.object(self.b,'publish_status'),TestClient(self.b.app) as client:
            with client.websocket_connect('/ws') as ws:
                ws.receive_json()
                ws.send_json(dict(type='start_person_search',description='red'))
                candidate=None
                while True:
                    m=ws.receive_json()
                    if m['type']=='candidate':candidate=m
                    if m['type']=='error':
                        self.assertEqual(m['code'],'crowd_failed');break
                self.assertTrue(self.b._awaiting_decision.is_set())
                ws.send_json(dict(type='confirm_candidate',candidate_id=candidate['candidate_id']))
                while True:
                    m=ws.receive_json()
                    if m['type']=='search_status' and m['state']=='confirmed':break
                done.set()
                while ws.receive_json().get('state')!='finished':pass
            self.b._run_thread.join(2)
            service.set_search_active.assert_any_call(True)
            service.set_search_active.assert_any_call(False)

if __name__=='__main__':unittest.main()
