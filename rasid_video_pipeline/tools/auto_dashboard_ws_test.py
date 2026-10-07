#!/usr/bin/env python3
"""Real dashboard WebSocket client, bounded live ROS test, scoring isolated here."""
import argparse
from functools import lru_cache
import yaml
import asyncio
import base64
import json
from pathlib import Path
import sys
import threading
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import cv2
import numpy as np
import websockets
from auto_live_test import Audit

@lru_cache(maxsize=1)
def configured_zones():
    path = Path(__file__).resolve().parents[2]/'simulation/aware_sim/config/zones.yaml'
    return {z['id']:z for z in yaml.safe_load(path.read_text())['zones']}

def validate(m):
    assert isinstance(m,dict) and 'sim_time' in m
    assert m['sim_time'] is None or isinstance(m['sim_time'],(int,float))
    t=m['type']
    fields={
        'search_status':['state','message','request_id'],
        'candidate':['candidate_id','track_id','frame_index','image','frame','justification'],
        'tracking_update':['candidate_id','track_id','frame_index','confirmed','image'],
        'person_location':['candidate_id','track_id','confirmed','location','drone','position_sim_time'],
        'rejected':['candidate_id','track_id','frame_index','status'],
        'lost':['candidate_id','track_id','frame_index','was_confirmed','status'],
        'error':['code','message','request_id','recoverable'],
        'gpt_verification_off':['code','message'],
        'crowd_analytics':['total_people_in_view','zones'],
        'crowd_snapshot':['image']}[t]
    assert all(f in m for f in fields),(t,fields)
    if t=='search_status':
        assert m['state'] in {'idle','searching','awaiting_decision','confirming','resuming_search','confirmed','finished'}
        assert isinstance(m['message'],str)
    if t in {'candidate','tracking_update','person_location','rejected','lost'}:
        assert isinstance(m['candidate_id'],str) and m['candidate_id']
        assert m['track_id'] is None or isinstance(m['track_id'],int)
    if 'frame_index' in m:assert isinstance(m['frame_index'],int)
    if 'confirmed' in m:assert isinstance(m['confirmed'],bool)
    if t=='error':assert isinstance(m['recoverable'],bool) and isinstance(m['message'],str)
    if t=='candidate':
        assert m['justification'] and 'confidence' not in m and 'score' not in m
    if t=='person_location':
        assert m['confirmed'] is True
        assert m['location'] is None or isinstance(m['position_sim_time'],(int,float))
        assert m['position_sim_time'] is None or m['position_sim_time'] <= m['sim_time']
        for f in ['location','drone']:
            assert m[f] is None or all(k in m[f] for k in ('lat','lon'))
    if t=='crowd_analytics':
        assert isinstance(m['total_people_in_view'],int) and m['total_people_in_view']>=0
        expected=set(configured_zones())
        assert set(m['zones'])==expected
        assert 'footfall' not in m and 'dwell' not in m
        for zone_id, zone in m['zones'].items():
            configured = configured_zones()[zone_id]
            for key in ('id','type','display_name','capacity'):
                assert zone[key] == configured[key], (zone_id, key)
            assert zone['booth'] == configured.get('booth')
            assert all(k in zone for k in ('count','capacity','coverage','coverage_fraction','occupancy_percent','crowded','last_seen_sim_time'))
            f=zone['coverage_fraction'];occ=zone['occupancy_percent']
            assert f is None or 0<=f<=1
            assert zone['capacity']>0
            assert zone['coverage'] in ('full','partial','unseen')
            assert zone['last_seen_sim_time'] is None or zone['last_seen_sim_time']<=m['sim_time']
            if f is not None and f>=.8:
                assert zone['coverage']=='full' and occ is not None
                assert abs(occ-100*zone['count']/zone['capacity'])<.001
                assert zone['crowded']==(occ>=80)
            else:
                assert occ is None and zone['crowded'] is None
                if f is None or f==0:assert zone['count'] is None and zone['coverage']=='unseen'
                else:assert zone['coverage']=='partial'
    for key in ['image','frame']:
        if key in m:
            assert m[key].startswith('data:image/jpeg;base64,')
            raw=base64.b64decode(m[key].split(',',1)[1],validate=True)
            image=cv2.imdecode(np.frombuffer(raw,dtype=np.uint8),cv2.IMREAD_COLOR)
            assert image is not None
            if t=='crowd_snapshot':assert image.shape[1]<=640

def validate_flow(messages):
    """Required healthy reject/confirm flow; conditional errors/loss tested separately."""
    states={m['state'] for m in messages if m['type']=='search_status'}
    assert {'searching','awaiting_decision','resuming_search','confirming','confirmed'} <= states, states
    candidates=[m for m in messages if m['type']=='candidate']
    assert len(candidates)>=2
    first,second=candidates[:2]
    assert first['candidate_id'] != second['candidate_id']
    rejection=next(m for m in messages if m['type']=='rejected')
    assert rejection['candidate_id']==first['candidate_id']
    ack=next(i for i,m in enumerate(messages) if m['type']=='search_status' and m['state']=='confirmed')
    for i,m in enumerate(messages):
        if m['type']=='person_location':
            assert i>ack and m['candidate_id']==second['candidate_id']
            assert m['location'] is not None and m['drone'] is not None
    assert any(m['type']=='tracking_update' and m['confirmed'] and m['candidate_id']==second['candidate_id'] for m in messages)


async def run(args,audit):
    counts={};errors=[];offers=[];rejected=False;confirmed=False;confirmed_updates=0
    received=[];crowd_received=[];snapshots=set();started_search=False
    search_start_wall=time.monotonic()+args.idle_observation_s
    pending=None;deadline=time.monotonic()+args.time_limit
    async with websockets.connect(args.url,max_size=8*1024*1024) as ws:
        await ws.send(json.dumps(dict(type='unknown_test_command',request_id='validation-probe')))
        while time.monotonic()<deadline:
            if not started_search and time.monotonic()>=search_start_wall:
                await ws.send(json.dumps(dict(type='start_person_search',request_id='live-start',description='a person wearing a red top and white trousers')))
                audit.write('SearchStarted',audit.sim_time)
                started_search=True
            if pending and time.monotonic()>=pending[0]:
                command=dict(type=pending[1],candidate_id=pending[2],request_id=pending[1])
                await ws.send(json.dumps(command));audit.write('Command',audit.sim_time,**command);pending=None
            try:m=json.loads(await asyncio.wait_for(ws.recv(),timeout=.2))
            except asyncio.TimeoutError:continue
            try:validate(m)
            except Exception as e:errors.append(f'Invalid message {m.get("type")}: {e}')
            received.append(m)
            t=m['type'];counts[t]=counts.get(t,0)+1
            if t=='crowd_analytics':crowd_received.append(dict(wall_time=time.time(),sim_time=m['sim_time'],during_search=started_search))
            if t=='crowd_snapshot':snapshots.add(m['sim_time'])
            clean=dict(m)
            for k in ['image','frame']:
                if k in clean:
                    name=f'{sum(counts.values()):04d}_{t}_{k}.jpg'
                    (args.log_dir/name).write_bytes(base64.b64decode(clean.pop(k).split(',',1)[1]))
                    clean[k+'_file']=name
            audit.write('WebSocketMessage',m['sim_time'],message=clean)
            if t=='candidate':
                offers.append(m['candidate_id'])
                if len(offers)>1 and not rejected:errors.append('Second candidate arrived before rejection acknowledgement')
                if len(offers)<=2:pending=(time.monotonic()+3,'reject_candidate' if len(offers)==1 else 'confirm_candidate',m['candidate_id'])
            if t=='rejected':rejected=True
            if t=='search_status' and m['state']=='confirmed':confirmed=True
            if t=='person_location':
                if not confirmed:errors.append('Location arrived before confirmation acknowledgement')
                loc=m['location'];ground=[loc['lat'],loc['lon']] if loc else None
                audit.write('PersonLocation',m['sim_time'],ground=ground,**audit.score(ground,m['position_sim_time']),position_sim_time=m['position_sim_time'],candidate_id=m['candidate_id'])
            if t=='tracking_update' and m['confirmed']:confirmed_updates+=1
            if t=='error' and m['request_id']!='validation-probe':errors.append(m['message'])
        for t in ['candidate','search_status','rejected','person_location','tracking_update','error']:
            if not counts.get(t):errors.append('Missing '+t)
        if len(offers)<2:errors.append('Fewer than two candidates')
        if not confirmed:errors.append('No actual confirmation acknowledgement')
        try:validate_flow(received)
        except Exception as e:errors.append('Flow validation: '+str(e))
    crowd_summary={}
    if args.with_crowd:
        if len(crowd_received)<4:errors.append('Too few crowd updates')
        if sum(not c['during_search'] for c in crowd_received)<2:errors.append('Crowd did not update regularly before search')
        if not any(c['during_search'] for c in crowd_received):errors.append('No crowd updates during search')
        missing=[c['sim_time'] for c in crowd_received if c['sim_time'] not in snapshots]
        if missing:errors.append('Crowd updates missing paired snapshots: '+str(missing))
        import statistics
        for label,items in [('all',crowd_received),('idle',[c for c in crowd_received if not c['during_search']]),('search',[c for c in crowd_received if c['during_search']])]:
            gaps=[b['wall_time']-a['wall_time'] for a,b in zip(items,items[1:])]
            crowd_summary[label]=dict(updates=len(items),mean_interval_s=statistics.mean(gaps) if gaps else None,median_interval_s=statistics.median(gaps) if gaps else None,max_interval_s=max(gaps) if gaps else None)
        if crowd_summary['all']['max_interval_s'] is not None and crowd_summary['all']['max_interval_s']>45:errors.append('Crowd update gap exceeded 45 wall seconds')
    result=dict(crowd=crowd_summary,passed=not errors,errors=errors,message_counts=counts,candidate_ids=offers,confirmed=confirmed,confirmed_tracking_updates=confirmed_updates,time_limit_s=args.time_limit)
    (args.log_dir/'ws_result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--time-limit',type=float,default=480);p.add_argument('--log-dir',type=Path,required=True);p.add_argument('--url',default='ws://127.0.0.1:8000/ws');p.add_argument('--with-crowd',action='store_true');p.add_argument('--idle-observation-s',type=float,default=0);args=p.parse_args()
    args.log_dir.mkdir(parents=True,exist_ok=True)
    import yaml
    import rclpy
    from rclpy.signals import SignalHandlerOptions
    from geometry_msgs.msg import PointStamped
    from rosgraph_msgs.msg import Clock
    venue=yaml.safe_load((Path(__file__).resolve().parents[2]/'simulation/aware_sim/config/zones.yaml').read_text())
    audit=Audit(args.log_dir,venue['venue']['location'])
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO);node=rclpy.create_node('dashboard_ws_test_scoring')
    node.create_subscription(PointStamped,'/aware/ground_truth/missing_person',audit.add_truth,10)
    def clock(m):audit.sim_time=m.clock.sec+m.clock.nanosec*1e-9
    node.create_subscription(Clock,'/clock',clock,10)
    stop=threading.Event()
    def spin():
        while not stop.is_set():rclpy.spin_once(node,timeout_sec=.1)
    thread=threading.Thread(target=spin,daemon=True);thread.start()
    try:result=asyncio.run(run(args,audit))
    finally:
        stop.set();thread.join(2);node.destroy_node();rclpy.shutdown();audit.finish()
    sys.exit(0 if result['passed'] else 1)
if __name__=='__main__':main()
