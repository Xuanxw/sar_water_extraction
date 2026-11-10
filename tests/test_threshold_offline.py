"""离线阈值：不重复滤波/归一化，保留负值，分块结果与整幅阈值一致。"""
import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin
from thres_segmentation.threshold import ThresholdSegmenter, read_input_preprocessing, percentile_mask
from thres_segmentation.scene_pipeline import run_scene_segmentation
from supervised.evaluation.boundary_eval import confusion_counts, boundary_regions


class OfflineThresholdTests(unittest.TestCase):
    def prepare(self, root):
        images, labels = root/'images/val', root/'labels/val'
        images.mkdir(parents=True); labels.mkdir(parents=True)
        name = 'BC3-synthetic.tif'
        profile = dict(driver='GTiff', height=23, width=29, count=1,
                       transform=from_origin(500000,3500000,3,3), crs='EPSG:32650')
        z = np.linspace(-2, 9, 23*29, dtype='float32').reshape(23,29)
        z[2,3] = np.nan; z[5,5] = 0
        truth = np.zeros(z.shape, dtype='uint8'); truth[:12,7:20]=1; truth[0]=255
        original = root/'original.tif'
        for path,data in ((images/name,z),(original,(z+3)*20)):
            with rasterio.open(path,'w',**profile,dtype='float32',nodata=np.nan) as f:f.write(data,1)
        with rasterio.open(labels/name,'w',**profile,dtype='uint8',nodata=255) as f:f.write(truth,1)
        row = dict(name=name,label_name=name,split='val',active=1,source_image=str(original))
        manifest=root/'split_manifest.csv'
        with manifest.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=row);w.writeheader();w.writerow(row)
            w.writerow(dict(row,name='must_not_read_train.tif',split='train'))
        receipt=dict(stage='normalized',complete=True,mean=[60.],std=[20.],
                     statistics_protocol='fixed_train_windows_only',
                     upstream=dict(stage='filtered',filter=dict(method='snap_lee',input_domain='amplitude',window_size=3,enl=1.)))
        (root/'preprocessing.json').write_text(json.dumps(receipt),encoding='utf-8')
        return images,labels,manifest,z,truth,receipt

    def test_full_scene_keeps_z_scores_and_is_block_independent(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary); images,labels,manifest,z,truth,receipt=self.prepare(root)
            valid=np.isfinite(z)
            for mode,percent in [('bottom',.5),('top',.5)]:
                expected,threshold=percentile_mask(z.astype("float64"),valid,percent,mode)
                for size in (7,13):
                    output=root/f'{mode}_{size}'
                    # 如果误入旧滤波分支，立即报错；不访问真实SNAP。
                    with patch('thres_segmentation.scene_pipeline.uniform_filter',side_effect=AssertionError('重复滤波')):
                        result=run_scene_segmentation(images,labels,manifest,output,
                            threshold_options=dict(preprocessing='offline_normalized',mode=mode,water_percent=percent),
                            block_size=size,max_visualizations=0,cache_dir=root/'cache')
                    with rasterio.open(output/'masks/threshold_VV/BC3-synthetic.tif') as f:
                        np.testing.assert_array_equal(f.read(1),np.where(valid,expected,255))
                    with (output/'per_image.csv').open(encoding='utf-8-sig') as f: row=next(csv.DictReader(f))
                    self.assertAlmostEqual(float(row['decision_threshold']),threshold)
                    self.assertEqual(row['threshold_domain'],'z_score')
                    self.assertEqual(int(row['image_valid_pixels']),int(valid.sum()))
                    self.assertEqual(json.loads((output/'input_preprocessing.json').read_text()),receipt)
                    counts=confusion_counts(truth,expected,valid&(truth!=255))
                    np.testing.assert_array_equal([result['threshold_VV'][k] for k in ('tn','fp','fn','tp')],counts)
                    with (output/'boundary/metrics.csv').open(encoding='utf-8-sig') as f: boundary=list(csv.DictReader(f))
                    for radius,areas in boundary_regions(truth,valid&(truth!=255)).items():
                        areas['boundary']=areas['inner']|areas['outer']
                        for region,area in areas.items():
                            row=next(r for r in boundary if int(r['radius_px'])==radius and r['region']==region)
                            np.testing.assert_array_equal([int(row[k]) for k in ('tn','fp','fn','tp')],confusion_counts(truth,expected,area))
            self.assertFalse(list((root/'cache').iterdir()))

    def test_input_contract_and_no_double_preprocessing(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary); *_,receipt=self.prepare(root)
            segmenter=ThresholdSegmenter(preprocessing='offline_normalized')
            read_input_preprocessing(root,segmenter,[1])
            with self.assertRaisesRegex(ValueError,'二次处理'):read_input_preprocessing(root,ThresholdSegmenter(),[1])
            with self.assertRaisesRegex(ValueError,'再次滤波'):ThresholdSegmenter(preprocessing='offline_normalized',lee_window=3)
            marker=root/'.preprocessing_incomplete';marker.touch()
            with self.assertRaisesRegex(ValueError,'尚未完成'):read_input_preprocessing(root,segmenter,[1])
            marker.unlink()
            for changes in ({'std':[0.]},{'statistics_protocol':'val_statistics'},{'mean':[]}):
                (root/'preprocessing.json').write_text(json.dumps(dict(receipt,**changes)))
                with self.assertRaisesRegex(ValueError,'训练集归一化统计'):read_input_preprocessing(root,segmenter,[1])

    def test_array_path_preserves_negative_zero_and_invalid(self):
        z=np.array([[-2.,0.,1.],[3.,np.nan,6.]])
        valid=np.isfinite(z)
        segmenter=ThresholdSegmenter(preprocessing='offline_normalized',water_percent=35)
        with patch('thres_segmentation.threshold.uniform_filter',side_effect=AssertionError('重复滤波')):
            predicted,score,threshold=segmenter.predict(z,valid)
        np.testing.assert_array_equal(score,z)
        expected,q=percentile_mask(z,valid,35)
        np.testing.assert_array_equal(predicted,expected)
        self.assertEqual(threshold,q)
