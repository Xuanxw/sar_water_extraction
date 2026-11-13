"""测试真正的C++可执行文件；解析公式与射线检查是独立参考。"""
import copy
import os
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from sar_geometry.core import solve_points, classify_profile

CORE = Path(os.environ.get('SAR_GEOMETRY_CORE',str(ROOT/'build/sar_geometry'/
            ('sar_geometry_core.exe' if os.name=='nt' else 'sar_geometry_core'))))


def straight_model():
    return dict(wavelength=0.056,range0=np.hypot(800000,100000),range_spacing=10,
                time0=0,line_time=0.01,columns=100000,rows=2001,look_side=-1,
                doppler_mode=0,doppler_sign=1,max_iterations=60,tolerance_hz=1e-6,
                tolerance_pixels=1e-7,max_step_seconds=20,
                orbit=np.array([[-20,7178137,-150000,0,0,7500,0],
                                [40,7178137,300000,0,0,7500,0]],dtype=float),
                doppler=np.empty((0,2)))


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CORE.is_file():
            raise RuntimeError('必须先构建sar_geometry_core，测试不静默跳过')

    def test_zero_coordinate_is_valid(self):
        r=solve_points(CORE,straight_model(),[[6378137,0,100000]],[[1,0,0]])[0]
        self.assertEqual(r[5],0)
        np.testing.assert_allclose(r[:3],0,atol=1e-8)
        self.assertAlmostEqual(r[6],np.degrees(np.arctan2(100000,800000)),places=8)

    def test_time_dependent_doppler_matches_independent_root(self):
        m=straight_model();m['doppler_mode']=1
        m['doppler']=np.array([[-20,0,60],[40,0,180]],float)
        p=np.array([6378137,75000,100000.])
        def residual(t):
            l=p-np.array([7178137,7500*t,0]);r=np.linalg.norm(l)
            return 2*7500*l[1]/(0.056*r)-(100+2*t)
        expected=brentq(residual,0,20,xtol=1e-13)
        r=solve_points(CORE,m,[p])[0]
        self.assertEqual(r[5],0)
        self.assertAlmostEqual(r[2],expected,places=9)
        self.assertLess(abs(residual(r[2])),1e-6)
        self.assertTrue(np.isnan(r[6]))

    def test_doppler_range_origins_and_sign(self):
        m=straight_model();m['doppler_mode']=1;m['doppler_sign']=-1
        m['doppler']=np.array([[-20,.004,90,30000],[40,.005,170,60000]],float)
        p=np.array([6378137,75000,100000.])
        def residual(t):
            l=p-np.array([7178137,7500*t,0]);r=np.linalg.norm(l);w=(t+20)/60
            fd=-((1-w)*(90+30000*(2*r/299792458-.004))+w*(170+60000*(2*r/299792458-.005)))
            return 2*7500*l[1]/(.056*r)-fd
        expected=brentq(residual,0,20)
        r=solve_points(CORE,m,[p])[0]
        self.assertEqual(r[5],0);self.assertAlmostEqual(r[2],expected,places=8)

    def test_sides_rotation_and_outside_coverage(self):
        m=straight_model();p=np.array([[6378137,75000,100000.]])
        reference=solve_points(CORE,m,p)
        angle=.73;rotation=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        m['orbit'][:,1:4]=m['orbit'][:,1:4]@rotation.T
        m['orbit'][:,4:7]=m['orbit'][:,4:7]@rotation.T
        rotated=solve_points(CORE,m,p@rotation.T)
        np.testing.assert_allclose(reference[:,:6],rotated[:,:6],atol=1e-7)
        m=straight_model();m['look_side']=1
        self.assertEqual(solve_points(CORE,m,p)[0,5],4)
        p[0,2]*=-1
        self.assertEqual(solve_points(CORE,m,p)[0,5],0)
        m['range0']+=100
        r=solve_points(CORE,m,p)[0];self.assertEqual(r[5],6);self.assertLess(r[0],0)

    def test_failures_are_not_valid(self):
        m=straight_model()
        self.assertEqual(solve_points(CORE,m,[[np.nan,0,1]])[0,5],1)
        m['max_iterations']=1
        self.assertEqual(solve_points(CORE,m,[[6378137,0,100000]])[0,5],3)
        m=straight_model();m['time0']=100
        self.assertEqual(solve_points(CORE,m,[[6378137,0,100000]])[0,5],2)
        m=straight_model();m['doppler_mode']=1;m['doppler']=np.array([[0,0,0],[1,0,0]],float)
        self.assertEqual(solve_points(CORE,m,[[6378137,0,100000]])[0,5],5)

    def test_quadratic_orbit_hermite(self):
        m=straight_model();m['doppler_mode']=1
        acceleration=2.0
        def orbit(t):
            return np.array([7178137+acceleration*t*t/2,7500*t,0]),np.array([acceleration*t,7500,0])
        m['orbit']=np.array([[t,*orbit(t)[0],*orbit(t)[1]] for t in (-20,0,20,40)])
        p=np.array([6378137,75000,100000.]);target=7.3
        s,v=orbit(target);l=p-s;fd=2*np.dot(v,l)/(.056*np.linalg.norm(l))
        m['doppler']=np.array([[-20,0,fd],[40,0,fd]])
        r=solve_points(CORE,m,[p])[0]
        self.assertEqual(r[5],0);self.assertAlmostEqual(r[2],target,places=8)


class VisibilityTests(unittest.TestCase):
    def test_ridge_blocks_front_facing_flat_target(self):
        x=np.array([0,200,500,800,1200]);z=np.array([0,300,0,0,0])
        r=np.hypot(x+1000,z-1000);angle=np.arctan2(z-1000,x+1000)
        result=classify_profile(CORE,np.column_stack((x,r,angle,np.ones(5))),True)
        self.assertEqual(result[2,0],1)
        self.assertEqual(result[3,0],0)
        self.assertLess(np.degrees(np.arctan2(1500,1000)),90)

    def test_envelope_against_independent_ray_segments(self):
        rng=np.random.default_rng(42);x=np.linspace(0,2000,100);z=rng.uniform(0,350,len(x));sensor=(-1000,1000)
        angles=np.arctan2(z-sensor[1],x-sensor[0]);ranges=np.hypot(z-sensor[1],x-sensor[0])
        actual=classify_profile(CORE,np.column_stack((x,ranges,angles,np.ones(len(x)))),True)[:,0]
        expected=[]
        for i in range(len(x)):
            ray=sensor[1]+(z[i]-sensor[1])*(x[:i]-sensor[0])/(x[i]-sensor[0])
            expected.append(int(np.any(z[:i]>ray+1e-7)))
        np.testing.assert_array_equal(actual,expected)

    def test_fold_expands_both_branches_and_allows_combined_flags(self):
        p=np.column_stack((np.arange(5),[1,4,2,3,5],[0,.3,.4,.2,.5],np.ones(5)))
        r=classify_profile(CORE,p,True)
        self.assertEqual(list(r[3]),[1,1])
        self.assertEqual(r[0,1],0);self.assertEqual(r[4,1],0)
        self.assertEqual(r[1,1],255);self.assertEqual(r[2,1],255)

    def test_unknown_support_holes_and_order(self):
        p=np.column_stack((np.arange(5),np.arange(1,6),np.arange(5)/10,np.ones(5)))
        np.testing.assert_array_equal(classify_profile(CORE,p,True),0)
        np.testing.assert_array_equal(classify_profile(CORE,p,False),255)
        p[2,3]=0;r=classify_profile(CORE,p,True)
        np.testing.assert_array_equal(r[2:,0],255);np.testing.assert_array_equal(r[:,1],255)
        with self.assertRaisesRegex(RuntimeError,'strictly increasing'):
            classify_profile(CORE,p[::-1],True)


if __name__=='__main__':
    unittest.main()
