"""无真值核查：抽样覆盖三层、未知区排除、分块风险一致、辅助缺失如实记录。"""
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import fiona
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from thres_segmentation.quality_review import export_quality_review

class QualityTests(unittest.TestCase):
    def write(self,path,data,transform=None):
        with rasterio.open(path,'w',driver='GTiff',count=1,dtype=data.dtype,
                height=data.shape[0],width=data.shape[1],nodata=255,crs='EPSG:32650',
                transform=transform or from_origin(500000,4500000,3,3)) as dst:dst.write(data,1)

    def test_sampling_and_block_consistency(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);data=np.zeros((41,53),np.uint8)
            data[5:34,8:42]=1;data[:2]=255;data[12:15,14:17]=255
            self.write(root/'mask.tif',data)
            reports=[];risk_masks=[];points=[]
            for name,block in (('a',11),('b',11),('c',80)):
                out=root/name
                report=export_quality_review(root/'mask.tif',out,points_per_stratum=9,
                        boundary_radius_px=3,block_size=block,log=lambda _:None)
                reports.append(report)
                self.assertEqual(report['water']['area_m2'],int((data==1).sum())*9)
                self.assertEqual(sum(v['population'] for v in report['sampling']['strata'].values()),int((data!=255).sum()))
                self.assertIsNone(report['accuracy']['f1'])
                self.assertIsNone(report['checks']['steep_water']['pixels'])
                self.assertEqual(set(fiona.listlayers(out/'review.gpkg')),{'verification_points','suspected_regions'})
                with fiona.open(out/'review.gpkg',layer='verification_points') as src:
                    rows=[dict(f['properties']) for f in src]
                self.assertEqual(len(rows),27)
                self.assertEqual(len({(r['row'],r['col']) for r in rows}),27)
                for r in rows:
                    self.assertEqual(r['pred'],int(data[r['row'],r['col']]))
                    self.assertIsNone(r['ref_class'])
                    self.assertAlmostEqual(r['incl_prob']*r['weight'],1.)
                points.append(rows)
                with fiona.open(out/'review.gpkg',layer='suspected_regions') as src:
                    geoms=[(f['geometry'],1) for f in src]
                risk_masks.append(rasterize(geoms,out_shape=data.shape,transform=from_origin(500000,4500000,3,3)))
            self.assertEqual(points[0],points[1])
            self.assertEqual(reports[0]['sampling']['strata'],reports[2]['sampling']['strata'])
            np.testing.assert_array_equal(risk_masks[0],risk_masks[2])
            self.assertEqual(reports[0]['checks'],reports[2]['checks'])

    def test_optional_auxiliary_and_empty_input(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);data=np.zeros((12,14),np.uint8);data[4:9,3:10]=1
            self.write(root/'mask.tif',data)
            self.write(root/'history.tif',np.zeros_like(data))
            self.write(root/'slope.tif',np.full(data.shape,20,np.float32))
            report=export_quality_review(root/'mask.tif',root/'out',history_path=root/'history.tif',
                    slope_path=root/'slope.tif',raw_mask_path=root/'mask.tif',log=lambda _:None)
            self.assertEqual(report['checks']['history_disagreement']['pixels'],35)
            self.assertEqual(report['checks']['steep_water']['pixels'],35)
            self.assertEqual(report['checks']['postprocess_changed']['pixels'],0)
            self.write(root/'empty.tif',np.full(data.shape,255,np.uint8))
            report=export_quality_review(root/'empty.tif',root/'empty',log=lambda _:None)
            self.assertEqual(report['coverage']['valid_pixels'],0)
            self.assertIsNone(report['water']['fraction_of_valid'])
            with fiona.open(root/'empty/review.gpkg',layer='verification_points') as src:self.assertEqual(len(src),0)

    def test_wrong_auxiliary_grid_rejected(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);data=np.zeros((8,8),np.uint8)
            self.write(root/'mask.tif',data)
            self.write(root/'history.tif',data,from_origin(1,1,3,3))
            with self.assertRaisesRegex(ValueError,'同网格'):
                export_quality_review(root/'mask.tif',root/'out',history_path=root/'history.tif')

if __name__=='__main__':unittest.main()
