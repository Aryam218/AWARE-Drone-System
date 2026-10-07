"""Evaluation-only inverse of aware_geo; no ground-truth input to inference."""
import math
import numpy as np
from cloud_track.foundation_model_wrappers import aware_geo as geo

def camera_basis(pose):
    """Camera right/down/forward axes in NED, derived from aware_geo rays."""
    h=float(pose['alt_rel_m'])
    def direction(u,v):
        point=geo.pixel_to_ground(u,v,pose)
        if point is None:raise ValueError('Calibration ray above horizon or invalid pose')
        north=(point[0]-pose['lat'])*geo.METERS_PER_DEG_LAT
        east=(point[1]-pose['lon'])*geo.METERS_PER_DEG_LAT*math.cos(math.radians(pose['lat']))
        ray=np.array([north,east,h],dtype=float)
        return ray/np.linalg.norm(ray)
    cx,cy=pose['cx'],pose['cy'];step=.25
    forward=direction(cx,cy)
    right=(direction(cx+step*pose['fx'],cy)*math.sqrt(1+step*step)-forward)/step
    down=(direction(cx,cy+step*pose['fy'])*math.sqrt(1+step*step)-forward)/step
    return np.column_stack((right,down,forward))

def ground_to_pixel(lat,lon,pose,basis=None,height_m=0.0):
    basis=camera_basis(pose) if basis is None else basis
    north=(lat-pose['lat'])*geo.METERS_PER_DEG_LAT
    east=(lon-pose['lon'])*geo.METERS_PER_DEG_LAT*math.cos(math.radians(pose['lat']))
    camera=np.linalg.solve(basis,np.array([north,east,pose['alt_rel_m']-height_m]))
    if camera[2]<=0:return None
    return (pose['cx']+pose['fx']*camera[0]/camera[2],pose['cy']+pose['fy']*camera[1]/camera[2],float(camera[2]))

def projected_people(people,pose,width,height,near=.1,far=80):
    basis=camera_basis(pose);out=[]
    for person in people:
        point=ground_to_pixel(person['lat'],person['lon'],pose,basis)
        inside=point is not None and 0<=point[0]<width and 0<=point[1]<height and near<=point[2]<=far
        out.append(dict(name=person['name'],pixel=list(point) if point else None,in_view=bool(inside)))
    return out

def interpolate_samples(samples,timestamp,max_gap):
    """Return neighboring samples and interpolation weight; reject stale references."""
    from bisect import bisect_left
    i=bisect_left([s['sim_time'] for s in samples],timestamp)
    if 0<i<len(samples):
        a,b=samples[i-1],samples[i]
        if max(timestamp-a['sim_time'],b['sim_time']-timestamp)>max_gap:raise ValueError('stale reference')
        f=(timestamp-a['sim_time'])/(b['sim_time']-a['sim_time']) if b['sim_time']>a['sim_time'] else 0
        return a,b,f
    a=samples[min(i,len(samples)-1)]
    if abs(a['sim_time']-timestamp)>max_gap:raise ValueError('stale reference')
    return a,a,0

def pose_at(samples,timestamp,lens):
    a,b,f=interpolate_samples(samples,timestamp,.5)
    pose={}
    for key in ('lat','lon','alt_rel_m','heading_deg','roll_deg','pitch_deg'):
        x,y=a['pose'].get(key,0),b['pose'].get(key,0)
        delta=(y-x+180)%360-180 if key in ('heading_deg','roll_deg','pitch_deg') else y-x
        pose[key]=x+f*delta
    pose.update(lens);return pose

def people_at(samples,timestamp):
    a,b,f=interpolate_samples(samples,timestamp,1.5)
    later={p['name']:p for p in b['people']}
    result=[]
    for p in a['people']:
        q=later[p['name']];result.append(dict(p,lat=p['lat']+f*(q['lat']-p['lat']),lon=p['lon']+f*(q['lon']-p['lon'])))
    return result
