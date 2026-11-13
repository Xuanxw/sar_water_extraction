"""XML契约和真正名义栅格入口测试；不访问原始数据，不运行生产分割。"""
import copy
from datetime import datetime,timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window
from pyproj import Transformer
import yaml

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sar_geometry.metadata import read_bc_metadata,make_model
from sar_geometry.terrain_io import surface_block,run_nominal
from sar_geometry.main import main
from sar_geometry.core import solve_points
CORE=Path(os.environ.get('SAR_GEOMETRY_CORE',str(ROOT/'build/sar_geometry'/('sar_geometry_core.exe' if os.name=='nt' else 'sar_geometry_core'))))


def xml_text(date='2025-01-01T00:00:00'):
    epoch=datetime.fromisoformat(date)
    stamps=[(epoch+timedelta(seconds=t)).isoformat() for t in (-10,10)]
    orbits=''.join(f'<orbit><time>{ts}</time><frame>Earth Fixed</frame><position><x>7178137</x><y>{7500*t}</y><z>0</z></position><velocity><x>0</x><y>7500</y><z>0</z></velocity></orbit>' for t,ts in zip((-10,10),stamps))
    doppler=''.join(f'<dcEstimate><azimuthTime>{ts}</azimuthTime><t0>0.004</t0><dataDcPolynomial>0 0 0</dataDcPolynomial></dcEstimate>' for ts in stamps)
    return f'''<product><adsHeader><productType>SLC</productType><missionId>S1B</missionId></adsHeader>
<generalAnnotation><productInformation><radarFrequency>5400000000</radarFrequency></productInformation><orbitList>{orbits}</orbitList></generalAnnotation>
<imageAnnotation><imageInformation><productFirstLineUtcTime>{date}</productFirstLineUtcTime><productLastLineUtcTime>{(epoch+timedelta(seconds=2)).isoformat()}</productLastLineUtcTime><look_side>left</look_side><slantRangeTime>0.004</slantRangeTime><rangePixelSpacing>10</rangePixelSpacing><azimuthTimeInterval>0.01</azimuthTimeInterval><numberOfSamples>100000</numberOfSamples><numberOfLines>201</numberOfLines></imageInformation></imageAnnotation>
<dopplerCentroid><dcEstimateList>{doppler}</dcEstimateList></dopplerCentroid></product>'''


def setup(root,rows=3):
    xml=root/'BC2-SM-SLC-test.xml';xml.write_text(xml_text(),encoding='utf-8')
    dem=root/'dem.tif'
    with rasterio.open(dem,'w',driver='GTiff',width=7,height=7,count=1,dtype='float32',crs='EPSG:4326',transform=from_origin(0,.1,.001,.001),nodata=-9999) as dst:
        data=np.zeros((7,7),np.float32);data[0,0]=-9999;dst.write(data,1)
    model=dict(radar_grid_confirmed=True,doppler_mode='zero',doppler_sign=None,height_datum='WGS84_ellipsoid',
               max_iterations=60,tolerance_hz=.00001,tolerance_pixels=1e-7,max_step_seconds=2)
    return dict(mode='nominal',paths=dict(XML_PATH=xml,DEM_PATH=dem,OUTPUT_DIR=root/'result',CORE_PATH=CORE),
                xml_member=None,model=model,execution=dict(block_rows=rows,resume=True))


class MetadataTests(unittest.TestCase):
    def test_exact_image_fields_and_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c=setup(root);p=c['paths']['XML_PATH']
            p.write_text(xml_text().replace('</product>','<unrelated><slantRangeTime>99</slantRangeTime></unrelated></product>'),encoding='utf-8')
            z=root/'source.zip'
            with ZipFile(z,'w') as archive:
                archive.writestr('BC2-SM-SLC/annotation/bc2-sm-slc.xml',p.read_bytes())
                archive.writestr('BC2-SM-SLC/annotation/calibration/ignored.xml','<x/>')
            m=read_bc_metadata(z)
            self.assertAlmostEqual(m['grid']['range0'],299792458*.004/2)
            self.assertEqual(m['orbit'].shape,(2,7));self.assertEqual(m['description']['product_type'],'SLC')
            self.assertEqual(m['description']['end_time_discrepancy_seconds'],0)

    def test_confirmation_and_leap_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=setup(Path(tmp));m=read_bc_metadata(c['paths']['XML_PATH']);settings=copy.deepcopy(c['model'])
            settings['radar_grid_confirmed']=False
            with self.assertRaisesRegex(ValueError,'radar_grid_confirmed'):make_model(m,settings)
            settings['radar_grid_confirmed']=True;settings['doppler_mode']=None
            with self.assertRaisesRegex(ValueError,'doppler_mode'):make_model(m,settings)
            settings['doppler_mode']='metadata';settings['doppler_sign']=None
            with self.assertRaisesRegex(ValueError,'Doppler符号'):make_model(m,settings)
            c['paths']['XML_PATH'].write_text(xml_text('2017-01-01T00:00:00'),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'闰秒'):read_bc_metadata(c['paths']['XML_PATH'])

    def test_surface_normal_and_nodata(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=setup(Path(tmp))
            with rasterio.open(c['paths']['DEM_PATH']) as src:
                t=Transformer.from_crs(4979,4978,always_xy=True)
                p,n=surface_block(src,Window(0,0,7,7),t)
            self.assertTrue(np.isnan(p[0,0]).all());self.assertTrue(np.isnan(n[0,3]).all())
            lat,lon=np.deg2rad(.0965),np.deg2rad(.0035)
            expected=[np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)]
            np.testing.assert_allclose(n[3,3],expected,atol=1e-7)


class NominalTests(unittest.TestCase):
    def test_full_entry_grid_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c=setup(root);path=root/'config.yaml'
            path.write_text(yaml.safe_dump(json.loads(json.dumps(c,default=str))),encoding='utf-8')
            report=main(path);self.assertEqual(sum(report['counts'].values()),49)
            self.assertEqual(report['counts']['invalid_input'],1)
            self.assertEqual(report['visibility'],'not_computed')
            with rasterio.open(root/'result/lookup.tif') as dst,rasterio.open(root/'dem.tif') as src:
                self.assertEqual((dst.shape,dst.transform,dst.crs),(src.shape,src.transform,src.crs))
                self.assertEqual(dst.count,2);self.assertTrue(np.isnan(dst.read()[:,0,0]).all())
            before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'result').glob('*.tif')}
            with patch('sar_geometry.terrain_io.solve_points',side_effect=AssertionError('must reuse')):main(path)
            after={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'result').glob('*.tif')}
            self.assertEqual(before,after)

    def test_interrupted_block_and_block_size_consistency(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c=setup(root,rows=2);metadata=read_bc_metadata(c['paths']['XML_PATH']);model=make_model(metadata,c['model'])
            calls=0
            def interrupted(*args,**kwargs):
                nonlocal calls
                calls+=1
                if calls==2:raise RuntimeError('simulated interruption')
                return solve_points(*args,**kwargs)
            with patch('sar_geometry.terrain_io.solve_points',side_effect=interrupted),self.assertRaisesRegex(RuntimeError,'simulated'):
                run_nominal(c,metadata,model,lambda x:None)
            self.assertFalse((root/'result/geometry_quality.json').exists())
            state=json.loads((root/'result/data_cache/progress.json').read_text())
            self.assertEqual(state['next_row'],2);self.assertFalse(state['complete'])
            report=run_nominal(c,metadata,model,lambda x:None)
            c['paths']['OUTPUT_DIR']=root/'other';c['execution']['block_rows']=7
            second=run_nominal(c,metadata,model,lambda x:None)
            self.assertEqual(report['counts'],second['counts'])
            for p in (root/'result').glob('*.tif'):
                with rasterio.open(p) as a,rasterio.open(root/'other'/p.name) as b:
                    np.testing.assert_allclose(a.read(),b.read(),rtol=0,atol=0,equal_nan=True)

    def test_no_silent_height_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=setup(Path(tmp));m=read_bc_metadata(c['paths']['XML_PATH']);model=make_model(m,c['model'])
            c['model']['height_datum']='EGM2008'
            with self.assertRaisesRegex(ValueError,'椭球高'):run_nominal(c,m,model)


if __name__=='__main__':unittest.main()
