"""Moving-camera snapshots: detector counts and coverage-aware venue zones.

No track IDs, footfall, dwell or ground-truth input. Coordinates use the flat
venue floor and body-fixed camera model from aware_geo.
"""
import math
import time
from pathlib import Path
import cv2
import numpy as np
import yaml
from PIL import Image
from cloud_track.foundation_model_wrappers import aware_geo as geo
from cloud_track.foundation_model_wrappers.aware_candidates import plausible
from analytics.shared_detector import shared_detector


def inside_polygon(x, y, polygon):
    inside = False
    for a, b in zip(polygon, polygon[1:]+polygon[:1]):
        ax, ay = a; bx, by = b
        cross = (x-ax)*(by-ay)-(y-ay)*(bx-ax)
        if abs(cross) < 1e-8 and min(ax,bx)-1e-8 <= x <= max(ax,bx)+1e-8 and min(ay,by)-1e-8 <= y <= max(ay,by)+1e-8:
            return True
        if (ay > y) != (by > y) and x < ax+(y-ay)*(bx-ax)/(by-ay):
            inside = not inside
    return inside


class MovingCrowdCounter:
    def __init__(self, detector=None, zones_path=None, sample_spacing_m=0.5):
        if sample_spacing_m <= 0:
            raise ValueError('sample spacing must be positive')
        if detector is None:
            from cloud_track.foundation_model_wrappers.grounding_dino_huggingface_wrapper import GroundingDinoHuggingfaceWrapper
            detector = GroundingDinoHuggingfaceWrapper(box_threshold=0.1, text_threshold=0.05)
        self.detector = shared_detector(detector)
        path = zones_path or Path(__file__).resolve().parents[2]/'simulation/aware_sim/config/zones.yaml'
        config = yaml.safe_load(Path(path).read_text())
        self.origin = config['venue']['location']
        self.zones = config['zones']
        self.last_seen = {z['id']: None for z in self.zones}
        self.samples = {}
        for zone in self.zones:
            polygon = zone['polygon']; a = np.asarray(polygon)
            points = [(x,y) for x in np.arange(a[:,0].min()+sample_spacing_m/2, a[:,0].max(), sample_spacing_m)
                      for y in np.arange(a[:,1].min()+sample_spacing_m/2, a[:,1].max(), sample_spacing_m)
                      if inside_polygon(x,y,polygon)]
            if not points:
                raise ValueError('zone has no coverage samples: '+zone['id'])
            self.samples[zone['id']] = np.array([self.world_to_geo(x,y) for x,y in points])

    def world_to_geo(self, x, y):
        # Same WGS84 local approximation as simulation patrol_plan. x=east, y=north.
        metres_per_degree = math.pi*6378137.0/180
        lat = self.origin['latitude']; lon = self.origin['longitude']
        return lat+y/metres_per_degree, lon+x/(metres_per_degree*math.cos(math.radians(lat)))

    def geo_to_world(self, lat, lon):
        metres_per_degree = math.pi*6378137.0/180
        return ((lon-self.origin['longitude'])*metres_per_degree*math.cos(math.radians(self.origin['latitude'])),
                (lat-self.origin['latitude'])*metres_per_degree)

    def zone_at(self, lat, lon):
        x, y = self.geo_to_world(lat, lon)
        return next((z['id'] for z in self.zones if inside_polygon(x,y,z['polygon'])), None)

    def coverage(self, pose, width, height):
        basis = geo.camera_basis(pose)
        result = {}
        for zone_id, points in self.samples.items():
            north = (points[:,0]-pose['lat'])*geo.METERS_PER_DEG_LAT
            east = (points[:,1]-pose['lon'])*geo.METERS_PER_DEG_LAT*math.cos(math.radians(pose['lat']))
            rays = np.column_stack((north,east,np.full(len(points),pose['alt_rel_m']))) @ basis
            depth = rays[:,2]
            with np.errstate(divide='ignore', invalid='ignore'):
                u = pose['cx']+pose['fx']*rays[:,0]/depth
                v = pose['cy']+pose['fy']*rays[:,1]/depth
            visible = (depth >= 0.1)&(depth <= 80)&(u >= 0)&(u < width)&(v >= 0)&(v < height)
            result[zone_id] = float(visible.mean())
        return result

    def count_detections(self, boxes, scores, pose, width, height, timestamp):
        boxes = np.asarray(boxes if boxes is not None else [], dtype=np.float32).reshape(-1,4)
        scores = np.asarray(scores if scores is not None else np.ones(len(boxes)), dtype=float)
        # The existing crowd class's IoU 0.4 NMS, followed by search plausibility.
        xywh = [[float(x),float(y),max(0.,float(x2-x)),max(0.,float(y2-y))] for x,y,x2,y2 in boxes]
        indices = cv2.dnn.NMSBoxes(xywh, scores.tolist(), 0.0, 0.4) if len(boxes) else []
        indices = np.asarray(indices if indices is not None else [], dtype=int).reshape(-1)
        nms_count = len(indices)
        indices = [i for i in indices if np.isfinite(boxes[i]).all() and boxes[i][2]>boxes[i][0] and boxes[i][3]>boxes[i][1] and plausible(boxes[i],width,height)]
        boxes = boxes[indices]
        valid_pose = pose is not None and all(pose.get(k) is not None for k in ('lat','lon','alt_rel_m','heading_deg','fx','fy','cx','cy')) and pose['alt_rel_m'] > 0.5
        coverage = self.coverage(pose,width,height) if valid_pose else {z['id']:None for z in self.zones}
        counts = {z['id']:0 if valid_pose else None for z in self.zones}
        people = []
        for box in boxes:
            ground = geo.box_to_ground(box,pose) if valid_pose else None
            zone = self.zone_at(*ground) if ground else None
            if zone is not None:
                counts[zone] += 1
            people.append(dict(box=box.tolist(),position=list(ground) if ground else None,zone=zone))
        zones = []
        for z in self.zones:
            key=z['id']; fraction=coverage[key]
            if fraction is not None and fraction > 0:
                self.last_seen[key] = timestamp
            occupancy = counts[key]/z['capacity']*100 if fraction is not None and fraction >= 0.8 and z['capacity'] > 0 else None
            zones.append(dict(id=key,type=z['type'],display_name=z['display_name'],booth=z.get('booth'),
                              visible_count=counts[key],capacity=z['capacity'],coverage_fraction=fraction,
                              last_seen=self.last_seen[key],occupancy_percent=occupancy))
        return dict(timestamp=timestamp,total_people=len(boxes),nms_detections=nms_count,people=people,zones=zones,
                    unlocated_people=sum(p['position'] is None for p in people))

    def analyze_frame(self, frame, pose=None, timestamp=None):
        timestamp = time.time() if timestamp is None else timestamp
        image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        _, _, boxes, scores = self.detector.run_inference(image,prompt='person',mark_results=False,print_results=False)
        height,width = frame.shape[:2]
        return self.count_detections(boxes,scores,pose,width,height,timestamp)
