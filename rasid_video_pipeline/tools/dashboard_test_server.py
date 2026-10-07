"""Test-only server instrumentation. No ground-truth input to inference."""
import argparse
import json
from pathlib import Path
import sys
import threading
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import backend
import pipeline_runner as runner
from auto_live_test import instrument_gpt_http
p=argparse.ArgumentParser();p.add_argument('--log-dir',type=Path,required=True);args=p.parse_args()
class Logger:
    sim_time=None
    lock=threading.Lock()
    def write(self,kind,timestamp=None,**data):
        with self.lock:
            with (args.log_dir/'server_events.jsonl').open('a') as f:
                f.write(json.dumps(dict(event=kind,sim_time=timestamp,wall_time=time.time(),**data),default=str)+'\n')
logger=Logger();instrument_gpt_http(logger)
# Measure actual allocations/inferences and process memory, not model estimates.
from cloud_track.foundation_model_wrappers.grounding_dino_huggingface_wrapper import GroundingDinoHuggingfaceWrapper
loaded=[]
original_init=GroundingDinoHuggingfaceWrapper.__init__
def observed_init(self,*a,**kw):
    original_init(self,*a,**kw)
    loaded.append(id(self))
    weights=sum(p.numel()*p.element_size() for p in self.model.parameters())
    logger.write('DetectorLoaded',None,detector_id=id(self),model_id=id(self.model),copies=len(loaded),parameter_bytes=weights,device=self.device)
GroundingDinoHuggingfaceWrapper.__init__=observed_init
original_inference=GroundingDinoHuggingfaceWrapper.run_inference
active=0;peak_active=0;inference_lock=threading.Lock()
def observed_inference(self,*a,**kw):
    global active,peak_active
    with inference_lock:active+=1;peak_active=max(peak_active,active)
    started=time.monotonic()
    mode='search' if backend._crowd is not None and backend._crowd.search_active.is_set() else 'idle'
    logger.write('DetectorStarted',logger.sim_time,detector_id=id(self),mode=mode,active=active)
    try:return original_inference(self,*a,**kw)
    finally:
        with inference_lock:active-=1
        logger.write('DetectorFinished',logger.sim_time,detector_id=id(self),mode=mode,duration_s=time.monotonic()-started,peak_active=peak_active)
GroundingDinoHuggingfaceWrapper.run_inference=observed_inference
original_publish=backend.publish_crowd
def observed_publish(analytics,image):
    logger.write('CrowdPublished',analytics['sim_time'],analytics=analytics,image_width=image.shape[1],image_height=image.shape[0])
    return original_publish(analytics,image)
backend.publish_crowd=observed_publish
memory_stop=threading.Event()
def sample_memory():
    while not memory_stop.is_set():
        fields={}
        for line in Path('/proc/self/status').read_text().splitlines():
            key,_,value=line.partition(':')
            if key in ('VmRSS','VmHWM'):fields[key+'_kib']=int(value.split()[0])
        try:
            for line in Path('/proc/self/smaps_rollup').read_text().splitlines():
                key,_,value=line.partition(':')
                if key=='Pss':fields['Pss_kib']=int(value.split()[0])
        except OSError:pass
        fields['mode']='startup' if backend._crowd is None else 'search' if backend._crowd.search_active.is_set() else 'idle'
        fields['detector_copies']=len(loaded)
        logger.write('MemorySample',logger.sim_time,**fields)
        memory_stop.wait(1)
threading.Thread(target=sample_memory,daemon=True).start()
original_controller=runner.SearchController
class ObservedController(original_controller):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        logger.write('SearchDetectorShared',None,detector_id=id(self.pipeline.backend.detector.detector),
                     shared_with_crowd=backend._crowd is not None and self.pipeline.backend.detector is backend._crowd.detector,
                     model_id=id(self.pipeline.backend.detector.model))
        original_redetect=self.pipeline._redetect
        def record_redetect(*a,**kw):
            forced=self.pipeline._force_redetect
            now=kw.get('now',a[-1] if a else logger.sim_time)
            logger.write('RedetectionStarted',now,forced=forced)
            result=original_redetect(*a,**kw)
            logger.write('RedetectionFinished',now,forced=forced,success=bool(result[1]),position_sim_time=self.pipeline._anchor_time)
            return result
        self.pipeline._redetect=record_redetect
        places=self.pipeline.backend.rejected_places
        add=places.add;check=places.is_rejected
        def record_add(ground,signature=None,now=None):
            add(ground,signature,now);logger.write('RejectedPlace',logger.sim_time,ground=ground)
        def record_check(ground,signature=None,now=None):
            result=check(ground,signature,now)
            if result:logger.write('RejectedPlaceSkip',logger.sim_time,ground=ground)
            return result
        places.add=record_add;places.is_rejected=record_check
    def process_frame(self,*a,**kw):
        logger.sim_time=kw.get('timestamp')
        event=super().process_frame(*a,**kw)
        if event is not None:
            target=self.pipeline.verified_target()
            logger.write(type(event).__name__,getattr(event,'timestamp',None),track_id=getattr(event,'track_id',None),
                ground=target['ground'] if target else None,position_sim_time=getattr(self.pipeline,'_anchor_time',None))
        return event
runner.SearchController=ObservedController
import uvicorn
try:uvicorn.run(backend.app,host='127.0.0.1',port=8000)
finally:memory_stop.set()
