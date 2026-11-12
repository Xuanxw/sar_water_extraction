"""整景阈值适配：全景百分位等价、跨块边界统计及同景原始XML追溯。"""
import csv
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from thres_segmentation.scene_pipeline import run_scene_segmentation, disk_percentiles
from thres_segmentation.threshold import ThresholdSegmenter
from thres_segmentation.metadata import read_source_metadata
from supervised.evaluation.boundary_eval import boundary_regions, confusion_counts


class BCThresholdTests(unittest.TestCase):
    def test_entry_dispatches_bc_single_vv(self):
        from thres_segmentation import main
        with patch("thres_segmentation.scene_pipeline.run_scene_segmentation", return_value={}) as run:
            main.main(Path(__file__).resolve().parents[1]/"configs/threshold_segmentation.yaml")
        self.assertEqual(run.call_args.kwargs['bands'], {'VV':1})
        self.assertEqual(run.call_args.kwargs['threshold_options']['preprocessing'],'offline_normalized')
        self.assertNotIn('stretch',run.call_args.kwargs['threshold_options'])
        config=main.load_yaml(Path(__file__).resolve().parents[1]/"configs/threshold_segmentation.yaml")
        self.assertEqual(run.call_args.kwargs['image_dir'],config['paths']['IMAGE_DIR'])
        self.assertEqual(run.call_args.kwargs['image_dir'],config['paths']['SPLIT_MANIFEST'].parent/'images/val')

    def test_disk_percentiles_match_numpy_with_ties(self):
        with TemporaryDirectory() as temp:
            path=Path(temp)/"values.bin"
            values=np.array([0.,0.,1.,2.,3.,8.,8.,100.])
            values.tofile(path)
            q=[0,1,10,50,99,100]
            np.testing.assert_allclose(disk_percentiles(path,len(values),q)[0],np.percentile(values,q))

    def test_custom_yaml_controls_all_dispatch_parameters(self):
        import yaml
        from thres_segmentation import main
        with TemporaryDirectory() as temp:
            config=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/threshold_segmentation.yaml').read_text(encoding='utf-8'))
            config['paths']['IMAGE_DIR']=str(Path(temp)/'images/val')
            config['paths']['LABEL_DIR']=None
            config['paths']['SPLIT_MANIFEST']=str(Path(temp)/'split_manifest.csv')
            config['paths']['REFERENCE_PATH']=str(Path(temp)/'history.tif')
            config['paths']['OUTPUT_DIR']=str(Path(temp)/'result')
            config['threshold']['water_percent']=7.5
            config['threshold']['mode']='top'
            config['input']['bands']={'VH':2}
            config['input']['scene_prefix']='BC4-'
            config['processing']['block_size']=2048
            config['quality_review']['points_per_stratum']=17
            config['output']['max_visualizations']=0
            path=Path(temp)/'test.yaml';path.write_text(yaml.safe_dump(config),encoding='utf-8')
            with patch('thres_segmentation.scene_pipeline.run_scene_segmentation',return_value={}) as run:
                main.main(path)
            args=run.call_args.kwargs
            self.assertEqual(args['image_dir'],Path(temp)/'images/val')
            self.assertIsNone(args['label_dir'])
            self.assertEqual(args['analysis_options']['reference_path'],Path(temp)/'history.tif')
            self.assertEqual(args['threshold_options']['water_percent'],7.5)
            self.assertEqual(args['threshold_options']['mode'],'top')
            self.assertEqual(args['bands'],{'VH':2})
            self.assertEqual(args['scene_prefix'],'BC4-')
            self.assertEqual(args['block_size'],2048)
            self.assertEqual(args['quality_review_options']['points_per_stratum'],17)
            self.assertEqual(args['max_visualizations'],0)
            self.assertEqual(args['output_dir'],Path(temp)/'result')

    def test_full_scene_matches_in_memory_and_boundary_counts(self):
        with TemporaryDirectory() as temp:
            root=Path(temp); images=root/'images/val';labels=root/'labels/val'
            images.mkdir(parents=True);labels.mkdir(parents=True)
            name='BC3-test.tif';rng=np.random.default_rng(42)
            data=rng.uniform(1,100,(39,47)).astype('float32');data[4,5]=0;data[3,7]=np.nan
            label=np.zeros(data.shape,dtype='uint8');label[8:30,10:35]=1;label[:2]=255
            profile=dict(driver='GTiff',height=39,width=47,count=1,transform=from_origin(100,200,3,3),crs='EPSG:32650')
            raw=root/'BC3-test.tiff'
            for path in (images/name,raw):
                with rasterio.open(path,'w',**profile,dtype='float32',nodata=np.nan) as dst:dst.write(data,1)
            with rasterio.open(labels/name,'w',**profile,dtype='uint8',nodata=255) as dst:dst.write(label,1)
            archive=root/'original.zip'
            xml=b'<product><polarisation>VV</polarisation><platformHeading>191.5</platformHeading><look_side>right</look_side></product>'
            with ZipFile(archive,'w') as z:
                z.write(raw,'folder/BC3-test.tiff');z.writestr('folder/BC3-test.xml',xml)
                z.writestr('unrelated.xml','<product><platformHeading>999</platformHeading></product>')
            manifest=root/'split_manifest.csv'
            record=dict(name=name,label_name=name,split='val',active=1,source_image=str(archive))
            with manifest.open('w',encoding='utf-8-sig',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=record);writer.writeheader();writer.writerow(record)
                # 这两条故意无实际文件，必须按清单排除而不是扫描目录或误读train。
                writer.writerow(dict(record,name="missing_train.tif",split="train"))
                writer.writerow(dict(record,name="inactive_val.tif",active=0))
            options=dict(water_percent=35,lee_window=5,noise_ratio=.12,stretch=(1,99),power_to_db=False)
            history = root/'history.tif'
            with rasterio.open(history,'w',**profile,dtype='uint8',nodata=255) as dst:
                dst.write(np.zeros(data.shape,dtype='uint8'),1)
            result=run_scene_segmentation(images,labels,manifest,root/'out',threshold_options=options,
                block_size=13,cache_dir=root/'cache',preview_size=20,max_visualizations=1,
                scene_analysis_options={name:dict(reference_path=history)},
                quality_review_options=dict(enabled=True,points_per_stratum=4,boundary_radius_px=2,block_size=13))
            self.assertTrue((root/'out/quality_review/threshold_VV/BC3-test/quality.json').is_file())
            valid=np.isfinite(data)
            expected,_,threshold=ThresholdSegmenter(**options).predict(data,valid)
            with (root/'out/analysis/threshold_VV/BC3-test/areas.csv').open(encoding='utf-8-sig') as stream:
                area_rows = {row['category']:row for row in csv.DictReader(stream)}
            self.assertAlmostEqual(float(area_rows['current_water']['m2']), int(expected.sum())*9)
            self.assertAlmostEqual(float(area_rows['current_valid']['m2']), int(valid.sum())*9)
            self.assertAlmostEqual(float(area_rows['new_water']['m2']), int(expected.sum())*9)
            with rasterio.open(root/'out/masks/threshold_VV'/name) as mask:
                np.testing.assert_array_equal(mask.read(1),np.where(valid,expected,255).astype('uint8'))
                self.assertEqual(mask.transform,profile['transform'])
            eval_valid=valid & (label!=255)
            np.testing.assert_array_equal([result['threshold_VV'][k] for k in ('tn','fp','fn','tp')],
                                          confusion_counts(label,expected,eval_valid))
            with (root/'out/boundary/metrics.csv').open(encoding='utf-8-sig') as f:
                rows=list(csv.DictReader(f))
            for radius,areas in boundary_regions(label,eval_valid).items():
                areas['boundary']=areas['inner']|areas['outer']
                for region,area in areas.items():
                    row=next(r for r in rows if int(r['radius_px'])==radius and r['region']==region)
                    np.testing.assert_array_equal([int(row[k]) for k in ('tn','fp','fn','tp')],confusion_counts(label,expected,area))
            with (root/'out/metadata.csv').open(encoding='utf-8-sig') as f: meta=next(csv.DictReader(f))
            self.assertEqual(meta['platformHeading'],'191.5')
            self.assertIn('!folder/BC3-test.xml',meta['source_xml'])
            self.assertEqual((root/'out/metadata/BC3-test.xml').read_bytes(),xml)
            self.assertTrue((root/'out/figures/BC3-test.png').is_file())
            self.assertFalse(list((root/'cache').iterdir()))
            # 翻转全部已知标签只改变精度；阈值及整幅预测必须不变。
            changed=np.where(label==255,255,1-label).astype('uint8')
            with rasterio.open(labels/name,'r+') as dst:dst.write(changed,1)
            run_scene_segmentation(images,labels,manifest,root/'out2',threshold_options=options,
                block_size=17,cache_dir=root/'cache',max_visualizations=0)
            with rasterio.open(root/'out2/masks/threshold_VV'/name) as mask:
                np.testing.assert_array_equal(mask.read(1),np.where(valid,expected,255).astype('uint8'))

    def test_unrelated_xml_is_not_associated(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);raw=root/'BC3-a.tif'
            with rasterio.open(raw,'w',driver='GTiff',width=3,height=3,count=1,dtype='float32',transform=from_origin(10,20,3,3)) as dst:
                dst.write(np.ones((3,3),dtype='float32'),1)
            (root/'BC3-b.xml').write_text('<product><platformHeading>999</platformHeading></product>')
            meta,_,xml=read_source_metadata(raw)
            self.assertEqual(meta['xml_status'],'missing');self.assertEqual(meta['platformHeading'],'');self.assertIsNone(xml)


if __name__=='__main__':
    unittest.main()
