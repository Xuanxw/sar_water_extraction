"""复核模式：只扫描不校准、Precision约束、待训练模型选择和单因素配置。"""
import csv
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT), str(ROOT/'tests')]
from scripts.models import calibrate_s1_unet_bn as experiment
import test_s1_bn_threshold_scan as bn_tests
from test_s1_bn_threshold_scan import small_model


class ReviewModeTests(unittest.TestCase):
    def test_script_directory_move_preserves_matching_protocol_only(self):
        moves = json.loads(Path(experiment.__file__).with_name('protocol_script_moves.json').read_text(encoding='utf-8'))
        old = dict(code={name: move['before'] for name, move in moves.items()}, batch_size=2)
        new = dict(code={move['path']: move['after'] for move in moves.values()}, batch_size=2)
        self.assertTrue(experiment.matches_calibration_protocol(old, new))
        self.assertFalse(experiment.matches_calibration_protocol(old, dict(new, batch_size=4)))
        new['code']['scripts/models/calibrate_bn.py'] = 'different source'
        self.assertFalse(experiment.matches_calibration_protocol(old, new))

    def test_thresholds_never_calibrates_or_reads_training_pixels_and_resumes(self):
        torch.set_num_threads(1)
        with TemporaryDirectory() as tmp:
            path, cfg = bn_tests.BNScanTests().prepare_run(Path(tmp))
            cfg['run']['MODE'] = 'thresholds'
            path.write_text(yaml.safe_dump(cfg),encoding='utf-8')
            hashes = {k:experiment.sha256(v) for k,v in cfg['paths']['CHECKPOINTS'].items()}
            original_read = experiment.WaterDataset.read_image
            read_names = []
            def read(data,index):
                name = data.image_paths[index].name
                read_names.append(name)
                self.assertIn(name, ('3.tif','4.tif'))
                return original_read(data,index)
            with patch.object(experiment,'load_model',side_effect=small_model), \
                    patch.object(experiment,'calibrate_layers',side_effect=AssertionError('不应校准BN')), \
                    patch.object(experiment.WaterDataset,'read_image',read):
                output = experiment.main(path)
            self.assertTrue(read_names)
            self.assertFalse(list(output.glob('*/calibrated.pth')))
            self.assertFalse((output/'bn_comparison.csv').exists())
            with (output/'summary.csv').open(encoding='utf-8-sig') as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),4)
            self.assertTrue(all(r['model'].endswith('_original') for r in rows))
            before=(output/'summary.csv').read_bytes()
            with patch.object(experiment,'load_model',side_effect=small_model), \
                    patch.object(experiment,'threshold_counts',side_effect=AssertionError('不应重复计数')):
                experiment.main(path)
            self.assertEqual(before,(output/'summary.csv').read_bytes())
            for alias,source in cfg['paths']['CHECKPOINTS'].items():
                self.assertEqual(hashes[alias],experiment.sha256(source))
            cfg['run']['MODE']='bn'
            path.write_text(yaml.safe_dump(cfg),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'计算协议已改变'):
                experiment.main(path)

    def test_precision_constraint_and_unattainable(self):
        # 三阈值下正类真值均为10、背景均为90；只有后两者满足P>=0.8。
        counts=np.array([[[80,10,0,10],[88,2,2,8],[90,0,7,3]]],dtype=np.int64)
        impossible=np.array([[[80,10,8,2],[85,5,9,1],[90,0,10,0]]],dtype=np.int64)
        with TemporaryDirectory() as tmp:
            output=Path(tmp)
            experiment.write_tables(output,dict(net_original=(['a'],counts),bad_original=(['a'],impossible)),[.1,.5,.9],.8)
            with (output/'precision_target.csv').open(encoding='utf-8-sig') as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(rows[0]['status'],'attainable')
            self.assertEqual(float(rows[0]['threshold']),.5)
            self.assertEqual(float(rows[0]['recall']),.8)
            self.assertEqual(rows[1]['status'],'unattainable')
            self.assertEqual(rows[1]['threshold'],'')
            self.assertEqual(rows[1]['recall'],'')

    def test_future_training_resolution_and_duplicate_rejection(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'train.yaml';path.write_text('model: {}',encoding='utf-8')
            cfg=dict(paths=dict(CHECKPOINTS={'fixed':Path('base.pth')},TRAIN_CONFIGS={'new':path}),
                     run=dict(EXPECTED_EPOCHS={'fixed':59}))
            with patch('scripts.models.predict_best_s1.candidate',return_value=(dict(checkpoint=str(Path(tmp)/'best.pth'),epoch=37),{})):
                selected=experiment.resolve_training_checkpoints(cfg)
            self.assertEqual(cfg['run']['EXPECTED_EPOCHS'],{'fixed':59,'new':37})
            self.assertEqual(selected['new']['config_sha256'],experiment.sha256(path))
            with self.assertRaisesRegex(ValueError,'别名.*重复'):
                experiment.resolve_training_checkpoints(cfg)

    def test_incomplete_training_errors_before_output_creation(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            train_cfg=dict(paths=dict(OUTPUT_DIR=str(root/'future')),run=dict(RESULT_FILES=dict(last='last.pth')),
                           model={},data=dict(DATA_LAYOUT='patch'),training=dict(EPOCHS=80))
            train_path=root/'train.yaml';train_path.write_text(yaml.safe_dump(train_cfg),encoding='utf-8')
            cfg=dict(paths=dict(TRAIN_CONFIGS={'future':str(train_path)},OUTPUT_DIR=str(root/'review')),
                     run=dict(MODE='thresholds'))
            path=root/'review.yaml';path.write_text(yaml.safe_dump(cfg),encoding='utf-8')
            with self.assertRaisesRegex(FileNotFoundError,'完整 80 轮'):
                experiment.main(path)
            self.assertFalse((root/'review').exists())

    def test_new_loss_configs_change_only_loss_and_output(self):
        folder=ROOT/'configs/s1'
        for model in ('pvt_deeplabv3plus','mobilenetv2_unet'):
            base=yaml.safe_load((folder/f'train_{model}_trainnorm80.yaml').read_text(encoding='utf-8'))
            new=yaml.safe_load((folder/f'train_{model}_trainnorm80_bce_dice.yaml').read_text(encoding='utf-8'))
            self.assertEqual(new['training']['EPOCHS'],80)
            self.assertIsNone(new['paths']['RESUME_CHECKPOINT'])
            self.assertEqual(new['training']['LOSS_NAME'],'bce_dice')
            self.assertEqual((new['training']['BCE_WEIGHT'],new['training']['DICE_WEIGHT']),(1.,1.))
            self.assertEqual(new['data']['PATCH_REPRESENTATION'],'vv_train_p02_p98_dem_m')
            self.assertNotEqual(new['paths']['OUTPUT_DIR'],base['paths']['OUTPUT_DIR'])
            new['paths']['OUTPUT_DIR']=base['paths']['OUTPUT_DIR']
            new['training']['LOSS_NAME']=base['training']['LOSS_NAME']
            self.assertEqual(new,base)
        scan=experiment.load_yaml(folder/'review_trainnorm80_thresholds.yaml')
        bn=experiment.load_yaml(folder/'review_trainnorm80_bn_unet.yaml')
        self.assertEqual(scan['run']['MODE'],'thresholds')
        self.assertEqual(sorted(scan['run']['EXPECTED_EPOCHS'].values()),[14,35,37,59,80])
        self.assertEqual(bn['run']['EXPECTED_EPOCHS'],dict(unet_best=35,unet_last=80))
        self.assertIsNone(bn['run']['CALIBRATION_SAMPLES'])


if __name__=='__main__':
    unittest.main()
