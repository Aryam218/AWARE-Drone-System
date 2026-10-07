#!/usr/bin/env python3
"""Bounded headless crowd evaluation. Only the scorer reads ground truth."""
import argparse,json,sys,threading,time,copy,signal
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import cv2,numpy as np
from crowd_projection import projected_people,pose_at,people_at

def main():
    p=argparse.ArgumentParser();p.add_argument('--log-dir',type=Path,required=True);p.add_argument('--frames',type=int,default=30);p.add_argument('--interval-s',type=float,default=12);p.add_argument('--time-limit',type=float,default=600);args=p.parse_args()
    args.log_dir.mkdir(parents=True,exist_ok=True)
    import rclpy
    from rclpy.signals import SignalHandlerOptions
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image,CameraInfo
    from std_msgs.msg import String
    from rosgraph_msgs.msg import Clock
    from ros_frame_source import ros_image_to_bgr
    from analytics.moving_crowd import MovingCrowdCounter
    lock=threading.RLock();stop=threading.Event();holder=dict(clock=None,image=None,lens=None,poses=[],truth=[])
    signal.signal(signal.SIGINT,lambda *_:stop.set());signal.signal(signal.SIGTERM,lambda *_:stop.set())
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO);node=rclpy.create_node('crowd_live_evaluation')
    posefile=(args.log_dir/'drone_poses.jsonl').open('w',buffering=1);truthfile=(args.log_dir/'ground_truth.jsonl').open('w',buffering=1)
    def clock(m):
        with lock:holder['clock']=m.clock.sec+m.clock.nanosec*1e-9
    def image(m):
        with lock:holder['image']=m
    def lens(m):
        with lock:holder['lens']=dict(fx=m.k[0],fy=m.k[4],cx=m.k[2],cy=m.k[5])
    def pose(m):
        data=json.loads(m.data)
        with lock:
            if holder['clock'] is None:return
            row=dict(sim_time=holder['clock'],pose=data)
            holder['poses'].append(row);posefile.write(json.dumps(row)+'\n')
    def truth(m):
        data=json.loads(m.data)
        with lock:holder['truth'].append(data);truthfile.write(json.dumps(data)+'\n')
    node.create_subscription(Clock,'/clock',clock,10)
    node.create_subscription(Image,'/aware/camera/image',image,qos_profile_sensor_data)
    node.create_subscription(CameraInfo,'/aware/camera/camera_info',lens,qos_profile_sensor_data)
    node.create_subscription(String,'/aware/drone/state',pose,10)
    node.create_subscription(String,'/aware/ground_truth',truth,10)
    def spin():
        while not stop.is_set() and rclpy.ok():rclpy.spin_once(node,timeout_sec=.1)
    thread=threading.Thread(target=spin,daemon=True);thread.start()
    started=time.monotonic();deadline=started+args.time_limit
    model_start=time.perf_counter();analytics=MovingCrowdCounter();model_load=time.perf_counter()-model_start
    detector_original=analytics.detector.run_inference;timing={}
    def timed(*a,**kw):
        t=time.perf_counter();kw['print_results']=False;result=detector_original(*a,**kw)
        timing.update(detector_s=time.perf_counter()-t,raw_detections=len(result[2]));return result
    analytics.detector.run_inference=timed
    print('MODEL_READY '+json.dumps(dict(model_load_s=model_load,device=analytics.detector.device)),flush=True)
    reused=MovingCrowdCounter(detector=analytics.detector)
    reuse=dict(same_detector=reused.detector is analytics.detector, same_model=reused.detector.model is analytics.detector.model)
    (args.log_dir/'reuse_check.json').write_text(json.dumps(reuse,indent=2))
    results=[];next_stamp=None;records=(args.log_dir/'frames.jsonl').open('w',buffering=1)
    try:
        while len(results)<args.frames and time.monotonic()<deadline and not stop.is_set():
            with lock:m=holder['image'];calibration=holder['lens']
            if m is None or calibration is None:time.sleep(.1);continue
            stamp=m.header.stamp.sec+m.header.stamp.nanosec*1e-9
            if next_stamp is not None and stamp<next_stamp:time.sleep(.05);continue
            frame=ros_image_to_bgr(m);i=len(results)+1;next_stamp=stamp+args.interval_s
            start=time.perf_counter()
            # The model receives ONLY the camera frame and its timestamp.
            from PIL import Image as PilImage
            _,_,boxes,scores=analytics.detector.run_inference(PilImage.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),prompt='person',mark_results=False,print_results=False)
            elapsed=time.perf_counter()-start
            with lock:poses=list(holder['poses']);truths=list(holder['truth'])
            try:
                frame_pose=pose_at(poses,stamp,calibration);people=people_at(truths,stamp)
                projected=projected_people(people,frame_pose,m.width,m.height)
            except (ValueError,IndexError,KeyError) as e:
                print('SCORING_UNAVAILABLE '+str(e),flush=True);continue
            # Only pose/calibration go to production counting. GT is scoring-only below.
            post_start=time.perf_counter()
            measured=analytics.count_detections(boxes,scores,frame_pose,m.width,m.height,stamp)
            post_s=time.perf_counter()-post_start
            count=measured['total_people']
            annotated=frame.copy()
            for person in measured['people']:
                x1,y1,x2,y2=map(int,person['box']);cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,255,0),2)
            by_name={p['name']:p for p in people}
            zone_truth={z['id']:0 for z in analytics.zones}
            for person in projected:
                if person['in_view']:
                    actor=by_name[person['name']];zone=analytics.zone_at(actor['lat'],actor['lon'])
                    if zone is not None:zone_truth[zone]+=1
            for z in measured['zones']:
                z['projected_ground_truth_count']=zone_truth[z['id']]
                z['error']=z['visible_count']-zone_truth[z['id']]
            reference=sum(p['in_view'] for p in projected)
            row=dict(frame=i,sim_time=stamp,analysis_s=elapsed+post_s,postprocess_s=post_s,**timing,people_count=int(count),nms_detections=measured['nms_detections'],zones=measured['zones'],detections=measured['people'],projected_ground_truth_count=reference,error=int(count)-reference,pose=frame_pose,projected_people=projected)
            results.append(row);records.write(json.dumps(row,default=lambda v:v.item() if hasattr(v,'item') else str(v))+'\n')
            cv2.imwrite(str(args.log_dir/f'{i:02d}_camera.jpg'),frame)
            cv2.imwrite(str(args.log_dir/f'{i:02d}_analytics.jpg'),annotated)
            overlay=frame.copy()
            for person in projected:
                if person['in_view']:
                    u,v,_=person['pixel'];cv2.circle(overlay,(int(u),int(v)),5,(0,255,255),2)
            cv2.putText(overlay,f'Projected GT feet: {reference}; analytics: {count}',(20,35),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,255),2)
            cv2.imwrite(str(args.log_dir/f'{i:02d}_projection.jpg'),overlay)
            print('FRAME '+json.dumps({k:v for k,v in row.items() if k not in ('pose','projected_people')}),flush=True)
            if i==1:
                before=time.perf_counter();other=reused.analyze_frame(frame,pose=frame_pose,timestamp=stamp)
                reuse.update(same_frame_boxes=measured['people']==other['people'],original_count=int(count),reused_count=int(other['total_people']),probe_s=time.perf_counter()-before,inference_calls=analytics.detector.inference_calls,cache_hits=analytics.detector.cache_hits)
                (args.log_dir/'reuse_check.json').write_text(json.dumps(reuse,indent=2))
    finally:
        stop.set();thread.join(2);node.destroy_node();rclpy.shutdown();records.close();posefile.close();truthfile.close()
    import statistics
    summary=dict(frames=len(results),requested_frames=args.frames,model_load_s=model_load,device=analytics.detector.device,mean_analysis_s=statistics.mean(r['analysis_s'] for r in results) if results else None,median_analysis_s=statistics.median(r['analysis_s'] for r in results) if results else None,mean_error=statistics.mean(r['error'] for r in results) if results else None,mae=statistics.mean(abs(r['error']) for r in results) if results else None,rmse=(statistics.mean(r['error']**2 for r in results)**.5) if results else None)
    summary['mean_postprocess_s']=statistics.mean(r['postprocess_s'] for r in results) if results else None
    summary['mean_detector_s']=statistics.mean(r['detector_s'] for r in results) if results else None
    summary['inference_calls']=analytics.detector.inference_calls
    summary['zones']={z['id']:dict(mae=statistics.mean(abs(next(q for q in r['zones'] if q['id']==z['id'])['error']) for r in results),mean_error=statistics.mean(next(q for q in r['zones'] if q['id']==z['id'])['error'] for r in results),seen_frames=sum(next(q for q in r['zones'] if q['id']==z['id'])['coverage_fraction']>0 for r in results),occupancy_frames=sum(next(q for q in r['zones'] if q['id']==z['id'])['occupancy_percent'] is not None for r in results)) for z in analytics.zones} if results else {}
    (args.log_dir/'summary.json').write_text(json.dumps(summary,indent=2));print('SUMMARY '+json.dumps(summary),flush=True)
    return 0 if len(results)==args.frames else 1
if __name__=='__main__':sys.exit(main())
