import unittest
import numpy as np
from core import upper_body_quality,GeometryQuality,ArmsCrossedRules,CutDetector
class Contracts(unittest.TestCase):
    def pose(self):
        p=np.full((17,2),100.,dtype=float)
        p[5:11]=[[80,70],[120,70],[65,100],[135,100],[90,120],[110,120]]
        return p
    def test_scores_do_not_gate(self):
        p=self.pose()
        for score in [0,.35,.5,7,float('nan')]:
            self.assertTrue(upper_body_quality(p,np.full(17,score),200,200)[0])
            self.assertTrue(GeometryQuality().update(0,p,np.full(17,score),200,200,[20,20,180,190])[0])
    def test_invalid_coordinates(self):
        p=self.pose();p[9,0]=float('nan')
        self.assertFalse(upper_body_quality(p,np.ones(17),200,200)[0])
    def test_fast_coherent_motion_not_rejected(self):
        q=GeometryQuality();p=self.pose();q.update(0,p,np.zeros(17),400,400,[0,0,399,399])
        ok,flags=q.update(.03,p+40,np.zeros(17),400,400,[0,0,399,399])
        self.assertTrue(ok);self.assertIn('large_coordinate_jump',flags)
    def test_cut_detection(self):
        c=CutDetector();self.assertFalse(c.update(np.zeros((40,40,3),np.uint8))[0])
        self.assertTrue(c.update(np.full((40,40,3),255,np.uint8))[0])
    def test_posture_segment_short_gap(self):
        p=self.pose();p[9]=[130,105];p[10]=[70,105]
        r=ArmsCrossedRules()
        for t in [0,.1,.2,.3,.4]:r.update(t,p)
        self.assertTrue(r.active);self.assertEqual(r.update(.5,None),[])
        events=r.update(.6,None);self.assertEqual(events[0]['end'],.4)
        self.assertFalse(r.active)
if __name__=='__main__':unittest.main()
