"""Protocol regression tests: mocked inference, real ASGI WebSocket/HTTP routes."""
import threading
import time
import unittest
from unittest.mock import patch
import numpy as np
from fastapi.testclient import TestClient
import backend as b
from pipeline_runner import CandidateEvent,ConfirmedEvent,RejectedEvent,LostEvent,VerificationErrorEvent

class ProtocolTests(unittest.TestCase):
    def setUp(self):
        startup_patch = patch.object(b, "create_crowd_service", return_value=None)
        startup_patch.start(); self.addCleanup(startup_patch.stop)
        b._crowd_latest = b._snapshot_latest = None
        b._status = dict(type="search_status", sim_time=None, state="idle", message="Ready", request_id=None)

    def test_socket_decisions_http_and_conditional_events(self):
        done=threading.Event()
        frame=np.zeros((64,64,3),dtype=np.uint8);box=np.array([5,5,30,50])
        def fake(**kw):
            self.assertEqual(kw["video_path"],"ros")
            kw['on_target_position'](dict(location={'lat':24.1,'lon':46.2},drone={'lat':24.3,'lon':46.4,'alt_rel_m':10,'heading_deg':0,'roll_deg':0,'pitch_deg':0},position_sim_time=10.0))
            kw['on_candidate'](CandidateEvent(frame,box,'Red top and white trousers',1,1,10.0))
            while not kw['should_reject'](): time.sleep(.005)
            kw['on_rejected'](RejectedEvent(2,11.0,1))
            kw['on_candidate'](CandidateEvent(frame,box,'Matches the description',2,3,12.0))
            while not kw['should_confirm'](): time.sleep(.005)
            kw['on_confirmed'](ConfirmedEvent(frame,box,2,4,13.0))
            kw['on_lost'](LostEvent(5,50.0,2,True))
            e=VerificationErrorEvent('OpenAI API has no credits: GPT verification is OFF',6,51.0)
            kw['on_verification_error'](e); kw['on_verification_error'](e)
            done.wait(3)
        with patch.object(b,'run_on_video',fake),patch.object(b,'publish_status'),TestClient(b.app) as client:
            with client.websocket_connect('/ws') as ws:
                self.assertEqual(ws.receive_json()['state'],'idle')
                ws.send_text('not JSON'); self.assertEqual(ws.receive_json()['code'],'invalid_request')
                ws.send_json({'type':'nonsense','request_id':'bad'}); self.assertEqual(ws.receive_json()['request_id'],'bad')
                ws.send_json({'type':'start_person_search','description':'red and white'})
                records=[]
                def until(kind,state=None):
                    while True:
                        m=ws.receive_json();records.append(m)
                        self.assertIn('sim_time',m)
                        if m['type']==kind and (state is None or m.get('state')==state):return m
                c1=until('candidate')
                self.assertTrue(c1['image'].startswith('data:image/jpeg;base64,'))
                self.assertNotIn('confidence',c1)
                ws.send_json({'type':'reject_candidate','candidate_id':'stale'})
                self.assertEqual(until('error')['code'],'command_rejected')
                ws.send_json({'type':'reject_candidate','candidate_id':c1['candidate_id']})
                until('rejected');c2=until('candidate')
                self.assertNotEqual(c1['candidate_id'],c2['candidate_id'])
                response=client.post('/confirm',json={'decision':'confirm'})
                self.assertEqual(response.json()['status'],'confirming_target')
                until('search_status','confirmed');loc=until('person_location')
                self.assertEqual(loc['location'],{'lat':24.1,'lon':46.2})
                self.assertEqual(loc['position_sim_time'],10.0)
                self.assertNotEqual(loc['location']['lat'],loc['drone']['lat'])
                self.assertTrue(until('lost')['was_confirmed'])
                until('gpt_verification_off');done.set();until('search_status','finished')
                self.assertEqual(sum(m['type']=='gpt_verification_off' for m in records),1)
            b._run_thread.join(3)
            self.assertIn('error',client.post('/confirm',json={'decision':'confirm'}).json())
            self.assertIn('error',client.post('/start',json={'video_path':'','description':'x'}).json())
    def test_runner_observes_verified_position_before_confirmation(self):
        from unittest.mock import Mock
        from types import SimpleNamespace
        import pipeline_runner as runner
        import drone_state
        frame=np.zeros((64,64,3),dtype=np.uint8)
        event=ConfirmedEvent(frame,np.array([5,5,30,50]),2,1,20.0)
        controller=Mock()
        controller.pipeline.backend.verification_error=None
        controller.pipeline._anchor_time=12.0
        controller.pipeline.verified_target.return_value={'ground':(24.1,46.2)}
        controller.process_frame.return_value=event
        pose=dict(lat=24.3,lon=46.4,alt_rel_m=10,heading_deg=0,roll_deg=1,pitch_deg=-8)
        reader=Mock();reader.frame_pose.return_value=pose
        class Stream:
            last_stamp=20.0
            def __iter__(self):return iter([frame])
            def release(self):pass
        observed=[]
        with patch.object(runner,'SearchController',return_value=controller),patch.object(runner,'open_stream',return_value=Stream()),patch.object(drone_state,'DroneStateReader',return_value=reader),patch.object(drone_state,'TargetPublisher'):
            runner.run_on_video('ros','person','red',Mock(),Mock(),Mock(),
                on_target_position=lambda value:observed.append(value),
                on_confirmed=lambda event:observed.append('acknowledgement'))
        self.assertEqual(observed[0]['location'],dict(lat=24.1,lon=46.2))
        self.assertEqual(observed[0]['drone'],pose)
        self.assertEqual(observed[0]['position_sim_time'],12.0)
        self.assertEqual(observed[1],'acknowledgement')

    def test_http_start_and_worker_failure(self):
        with patch.object(b,'run_on_video',side_effect=RuntimeError('camera failed')),patch.object(b,'publish_status'),TestClient(b.app) as client:
            with client.websocket_connect('/ws') as ws:
                ws.receive_json()
                self.assertTrue(client.post('/start',json={'video_path':'ros','description':'x'}).json()['ok'])
                while True:
                    m=ws.receive_json()
                    if m['type']=='error': break
                self.assertEqual(m['code'],'search_failed');self.assertFalse(m['recoverable'])
            b._run_thread.join(3)

if __name__=='__main__':unittest.main()
