"""方向修正后的GUI共用判据，包含跨北、左右视及DEM空值。"""
import sys,unittest,textwrap
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from thres_segmentation.terrain_geometry import classify_local_radar_terrain,geometry_parameters
class TerrainTests(unittest.TestCase):
    def test_gui_shared_classification_and_unknown(self):
        source=(ROOT/'src/thres_segmentation/gis_worker.py').read_text(encoding='utf-8')
        code=textwrap.dedent(source[source.index('                # 与逐景阈值入口共用方向修正后的公式'):source.index('                aligned_cls = classification')])
        rng=np.random.default_rng(42);s=rng.uniform(0,90,(101,113)).astype('float32');a=rng.uniform(0,360,s.shape).astype('float32')
        s[0,:]=np.nan;a[:,0]=np.nan
        for heading in (0,192.635096,347.37916,359):
            for look in ('left','right'):
                env=dict(np=np,alpha=heading,theta=40.3748,look_dir=look,aligned_slope=s,aligned_aspect=a,
                    aligned_slope_valid=np.isfinite(s).astype('uint8'),aligned_aspect_valid=np.isfinite(a).astype('uint8'),
                    sar_shape=s.shape,cfg={'slope_thresh':15})
                exec(code,env)
                np.testing.assert_array_equal(classify_local_radar_terrain(s,a,heading,40.3748,look,15),env['classification'])
    def test_classes_and_missing_metadata(self):
        s=np.array([[0,20,50,60,np.nan]],np.float32);a=np.array([[270,270,270,90,270]],np.float32)
        self.assertEqual(classify_local_radar_terrain(s,a,0,40,'right',15).tolist(),[[4,3,2,1,0]])
        with self.assertRaises(KeyError):geometry_parameters({})
        with self.assertRaises(ValueError):geometry_parameters(dict(platformHeading='0',incidenceAngleMidSwath='40',look_side='unknown'))
if __name__=='__main__':unittest.main()
