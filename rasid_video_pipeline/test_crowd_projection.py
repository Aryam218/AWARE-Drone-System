import unittest
import numpy as np
from tools.crowd_projection import ground_to_pixel,projected_people,pose_at
from cloud_track.foundation_model_wrappers import aware_geo as geo
class ProjectionTests(unittest.TestCase):
    def test_inverse_of_aware_geo_with_attitude(self):
        for heading,roll,pitch in [(0,0,0),(123,12,-8),(280,-15,9)]:
            pose=dict(lat=24.7136,lon=46.6753,alt_rel_m=10,heading_deg=heading,roll_deg=roll,pitch_deg=pitch,fx=600,fy=600,cx=640,cy=480)
            for u,v in [(640,480),(100,200),(1000,800)]:
                point=geo.pixel_to_ground(u,v,pose)
                np.testing.assert_allclose(ground_to_pixel(*point,pose)[:2],(u,v),atol=.001)
    def test_bounds_and_behind_camera(self):
        pose=dict(lat=24,lon=46,alt_rel_m=10,heading_deg=0,fx=600,fy=600,cx=640,cy=480)
        inside=geo.pixel_to_ground(640,480,pose);outside=geo.pixel_to_ground(1500,480,pose)
        people=[dict(name='inside',lat=inside[0],lon=inside[1]),dict(name='outside',lat=outside[0],lon=outside[1]),dict(name='behind',lat=24-50/geo.METERS_PER_DEG_LAT,lon=46)]
        self.assertEqual([r['in_view'] for r in projected_people(people,pose,1280,960)],[True,False,False])
    def test_pose_interpolation_wrap_and_stale_rejection(self):
        a=dict(lat=24,lon=46,alt_rel_m=10,heading_deg=359,roll_deg=0,pitch_deg=0)
        b=dict(a,heading_deg=1)
        samples=[dict(sim_time=0,pose=a),dict(sim_time=.2,pose=b)]
        self.assertAlmostEqual(pose_at(samples,.1,{})['heading_deg'],360)
        with self.assertRaises(ValueError):pose_at(samples,2,{})
if __name__=='__main__':unittest.main()
