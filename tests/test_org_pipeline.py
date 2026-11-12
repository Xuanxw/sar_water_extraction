"""ORG生产合成验证：不调用SNAP、不下载辅助数据、不使用真实密云影像。"""
import copy
import importlib.util
import json
import shutil
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zipfile import ZipFile

import fiona
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from shapely.geometry import box, mapping, shape

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from configuration import load_yaml
from thres_segmentation import org_pipeline as pipeline
from thres_segmentation.org_io import prepare_auxiliary, read_aoi, inventory


class OrgProductionTests(unittest.TestCase):
    def make_config(self,root):
        config=load_yaml(ROOT/'configs/北京密云区/threshold_segmentation.yaml')
        paths=config['paths']
        for key,name in [('ORG_ROOT','ORG'),('OUTPUT_DIR','out'),('CACHE_DIR','cache'),('AUXILIARY_CACHE_DIR','auxcache')]:
            paths[key]=root/name
        paths['ORG_ROOT'].mkdir()
        paths['AOI_PATH']=root/'aoi.shp'
        self.aoi=box(500005,4000005,500055,4000055)
        with fiona.open(paths['AOI_PATH'],'w',driver='ESRI Shapefile',crs='EPSG:32650',
                        schema={'geometry':'Polygon','properties':{'id':'int'}}) as dst:
            dst.write({'geometry':mapping(self.aoi),'properties':{'id':1}})
        profile=dict(driver='GTiff',width=64,height=64,count=1,dtype='float32',nodata=np.nan,
                     crs='EPSG:32650',transform=from_origin(500000,4000064,1,1))
        data=np.arange(4096,dtype='float32').reshape(64,64)+1
        data[20:27,30:40]=1;data[24,35]=np.nan
        self.data=data;self.profile=profile
        for sensor in ('BC2','BC3'):
            name=f'{sensor}-SM-ORG-VV-20260101T030000-000001'
            tif=paths['ORG_ROOT']/(name+'.tif')
            with rasterio.open(tif,'w',**profile) as dst:dst.write(data,1)
            xml='<product><polarisation>VV</polarisation></product>'
            if sensor=='BC3':
                with ZipFile(paths['ORG_ROOT']/(name+'.zip'),'w') as z:
                    z.write(tif,name+'.tif');z.writestr(name+'.xml',xml)
                tif.unlink()
            else:tif.with_suffix('.XML').write_text(xml,encoding='utf-8')
        # 不完整下载/SLC不在清单中。
        (paths['ORG_ROOT']/'BC2-SM-SLC-VV-20260101T030000-000001.zip').write_bytes(b'not selected')
        cover=np.full((64,64),40,np.float32);cover[12:20,10:40]=50;cover[24:30,20:45]=80
        for key,name,values in [('LANDCOVER_PATH','cover.tif',cover),('ELEVATION_PATH','dem.tif',np.zeros((64,64),np.float32))]:
            paths[key]=root/name
            with rasterio.open(paths[key],'w',**profile) as dst:dst.write(values,1)
        options=config['org'];options.update(crop_buffer_m=2,save_quicklook=True)
        options['filter']['method']='none'
        options['auxiliary'].update(download_missing=False,buffer_m=2,landcover_resolution_m=1,dem_resolution_m=1)
        config['processing']['block_size']=13
        config['output']['preview_size']=64
        config['quality_review'].update(points_per_stratum=4,boundary_radius_px=2,block_size=13)
        return config

    def test_production_resume_and_optional_merge(self):
        with TemporaryDirectory() as temp:
            config=self.make_config(Path(temp))
            with patch('thres_segmentation.org_io.download_tile',side_effect=AssertionError('no download')):
                root=pipeline.run_org_scenes(config)
            rows=json.loads((root/'results.json').read_text())
            self.assertEqual(len(rows),2)
            self.assertFalse((root/'2026-01-01/water.shp').exists())
            for row in rows:
                folder=Path(row['water_shp']).parent
                report=json.loads((folder/'quality.json').read_text(encoding='utf-8'))
                self.assertIsNone(report['accuracy']['f1'])
                self.assertEqual(report['checks']['history_disagreement']['status'],'computed')
                self.assertEqual(report['checks']['steep_water']['status'],'computed')
                self.assertTrue((folder/report['artifacts']['vectors']).is_file())
                with rasterio.open(folder/'water.tif') as src:
                    values=src.read(1);inside=geometry_mask([mapping(self.aoi)],out_shape=src.shape,transform=src.transform,invert=True)
                    self.assertTrue(np.all(values[~inside]==255))
                    self.assertTrue(set(np.unique(values)) <= {0,1,255})
                    area=float((values==1).sum())
                with fiona.open(row['water_shp']) as src:
                    polygons=[shape(f['geometry']) for f in src]
                    self.assertTrue(all(p.difference(self.aoi).area==0 for p in polygons))
                    self.assertAlmostEqual(sum(p.area for p in polygons),area)
                self.assertTrue((folder/'quicklook.png').exists())
            # 修改显示设置不能导致已有水体产品重算。
            config['logging']['console_color']=False
            before={p:p.stat().st_mtime_ns for row in rows for p in Path(row['water_shp']).parent.rglob('*') if p.is_file()}
            with patch.object(pipeline,'crop_org',side_effect=AssertionError('crop repeated')),patch.object(pipeline,'segment',side_effect=AssertionError('segment repeated')):
                pipeline.run_org_scenes(config)
            self.assertEqual(before,{p:p.stat().st_mtime_ns for p in before})
            spec=importlib.util.spec_from_file_location('merge_water',ROOT/'scripts/merge_water_shp.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            merged=module.merge_results(root,Path(temp)/'merged','EPSG:32650')
            with fiona.open(merged/'2026-01-01_water.shp') as src:
                self.assertAlmostEqual(sum(shape(f['geometry']).area for f in src),2*area)  # 两景重叠面保留

    def test_interrupted_after_masks_resumes_without_resegmentation(self):
        with TemporaryDirectory() as temp:
            config=self.make_config(Path(temp))
            with patch.object(pipeline,'vectorize',side_effect=RuntimeError('simulated interruption')):
                with self.assertRaisesRegex(RuntimeError,'simulated interruption'):pipeline.run_org_scenes(config)
            masks=list(config['paths']['OUTPUT_DIR'].glob('*/*/water.tif'))
            self.assertEqual(len(masks),1)
            first=masks[0].parent;stamp=(first/'water.tif').stat().st_mtime_ns
            segment=pipeline.segment
            def guarded(*args,**kwargs):
                self.assertNotEqual(args[4],first,'Completed mask must be reused')
                return segment(*args,**kwargs)
            with patch.object(pipeline,'segment',side_effect=guarded):pipeline.run_org_scenes(config)
            self.assertEqual((first/'water.tif').stat().st_mtime_ns,stamp)
            state=json.loads((config['paths']['CACHE_DIR']/'scenes'/first.name/'progress.json').read_text(encoding='utf-8'))
            self.assertTrue(state['stages']['complete']['complete'])

    def test_cache_and_results_can_be_relocated(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);config=self.make_config(root)
            pipeline.run_org_scenes(config)
            old_output=config['paths']['OUTPUT_DIR'];old_cache=config['paths']['CACHE_DIR']
            new_output=root/'relocated/results';new_cache=root/'relocated/data_cache'
            shutil.copytree(old_output,new_output);shutil.copytree(old_cache,new_cache)
            config['paths'].update(OUTPUT_DIR=new_output,CACHE_DIR=new_cache)
            # 模拟旧run.json遗留路径与key；它仅作运行记录，不能阻止迁移。
            (new_output/'run.json').write_text('{"key":"old-path-signature","config":{}}',encoding='utf-8')
            before={p:p.stat().st_mtime_ns for p in new_output.glob('*/*/water.*')}
            with patch.object(pipeline,'crop_org',side_effect=AssertionError('crop after relocation')), \
                 patch.object(pipeline,'segment',side_effect=AssertionError('segment after relocation')):
                pipeline.run_org_scenes(config)
            self.assertEqual(before,{p:p.stat().st_mtime_ns for p in before})
            records=json.loads((new_output/'results.json').read_text(encoding='utf-8'))
            self.assertTrue(all(Path(r['water_shp']).is_relative_to(new_output) for r in records))
            for receipt in new_cache.glob('scenes/*/progress.json'):
                state=json.loads(receipt.read_text(encoding='utf-8'))
                for stage in state['stages'].values():
                    self.assertTrue(all(not Path(f['path']).is_absolute() for f in stage['files']))

    def test_resume_false_reprocesses_scenes(self):
        with TemporaryDirectory() as temp:
            config=self.make_config(Path(temp));pipeline.run_org_scenes(config)
            config['org']['resume']=False
            with patch.object(pipeline,'segment',wraps=pipeline.segment) as segment:
                pipeline.run_org_scenes(config)
            self.assertEqual(segment.call_count,2)

    def test_optional_auxiliary_paths_can_be_omitted_or_null(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);config=self.make_config(root)
            config['org']['auxiliary']['download_missing']=True
            aoi,crs=read_aoi(config['paths']['AOI_PATH'],'EPSG:32650')
            local={'worldcover':root/'cover.tif','copernicus':root/'dem.tif'}
            for omit in (True,False):
                with self.subTest(omit=omit):
                    for key in ('LANDCOVER_PATH','ELEVATION_PATH','ORG_HISTORY_PATH'):
                        if omit:config['paths'].pop(key,None)
                        else:config['paths'][key]=None
                    with patch('thres_segmentation.org_io.tile_specs',side_effect=lambda product,*args: [
                                  {'url':product,'filename':product+'.tif'}]), \
                         patch('thres_segmentation.org_io.download_tile',side_effect=lambda url,*args,**kwargs:local[url]) as download:
                        result=prepare_auxiliary(config,aoi,crs)
                    self.assertEqual(download.call_count,2)
                    self.assertTrue(all(result[k].is_file() for k in ('worldcover','copernicus','slope')))
            config['org']['auxiliary']['download_missing']=False
            with self.assertRaisesRegex(FileNotFoundError,'禁止下载'):
                prepare_auxiliary(config,aoi,crs)

    def test_morphology_matches_single_block(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);config=self.make_config(root)
            config['org']['postprocess'].update(opening_radius_px=1,closing_radius_px=2)
            aoi,crs=read_aoi(config['paths']['AOI_PATH'],'EPSG:32650')
            auxiliary=prepare_auxiliary(config,aoi,crs)
            source=next(config['paths']['ORG_ROOT'].glob('*.tif'))
            small=root/'small';large=root/'large';small.mkdir();large.mkdir()
            stats=pipeline.statistics(source,aoi,config,small)
            pipeline.segment(source,aoi,auxiliary,config,small,stats)
            config['processing']['block_size']=128
            pipeline.segment(source,aoi,auxiliary,config,large,stats)
            with rasterio.open(small/'water.tif') as a,rasterio.open(large/'water.tif') as b:
                np.testing.assert_array_equal(a.read(1),b.read(1))
            changed=copy.deepcopy(config);changed['org']['auxiliary']['elevation_scale']=2
            aux2=prepare_auxiliary(changed,aoi,crs)
            self.assertEqual(auxiliary['worldcover'],aux2['worldcover'])
            self.assertEqual(auxiliary['copernicus'],aux2['copernicus'])
            self.assertNotEqual(auxiliary['slope'],aux2['slope'])

    def test_inventory_is_dynamic_and_entry_uses_case_yaml(self):
        from thres_segmentation import main
        self.assertTrue(main.CONFIG_PATH.is_file())
        with TemporaryDirectory() as temp:
            config=self.make_config(Path(temp))
            records=inventory(config['paths']['ORG_ROOT'],config['org'])
            self.assertEqual(len(records),2)
            self.assertTrue(all('-ORG-' in r['scene_id'] for r in records))
            (config['paths']['ORG_ROOT']/next(r['path'] for r in records if r['sensor']=='BC3')).unlink()
            self.assertEqual(len(inventory(config['paths']['ORG_ROOT'],config['org'])),1)
        with patch.object(pipeline,'run_org_scenes',return_value='selected') as run:
            self.assertEqual(main.main(),'selected')
            chosen=run.call_args.args[0]
            self.assertEqual(chosen['run']['mode'],'org_scenes')
            self.assertNotIn('expected_scenes',chosen['org'])
            self.assertNotIn('expected_dates',chosen['org'])


if __name__=='__main__':unittest.main()
