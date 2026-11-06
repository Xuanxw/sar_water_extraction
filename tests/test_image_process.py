"""离线滤波、归一化、快视图、训练输入及续跑，只用合成数据与模拟GPT。"""
import json
import hashlib
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
import rasterio
import torch
from datasets import image_process as tools, snap_preprocess
from supervised.load.sar_scene_dataset import SARSceneDataset, check_preprocessed_input
from supervised.train import WaterSegmentationTrainer
from supervised.evaluation import model_eval
import test_sar_scene_dataset as fixture

class ImageProcessTests(unittest.TestCase):
    def options(self, root):
        return dict(OUTPUT_DIR=root/'raw',CALIBRATED_ROOT=root/'processed/calibrated',
                    FILTERED_ROOT=root/'processed/filtered',NORMALIZED_ROOT=root/'processed/normalized',
                    WINDOW_CACHE_DIR=root/'windows',SNAP_CACHE_DIR=root/'snap',
                    WINDOW_SIZE=32,HALO=8,NORMALIZATION_WINDOWS=2,SEED=42,BANDS=(1,),
                    ENABLE_CALIBRATION=False,ENABLE_FILTER=True,FILTER_INPUT_STAGE='org',NORMALIZE_INPUT_STAGE='filtered')

    def fake_gpt(self,source,bands,output,work,diagnostics):
        shutil.copyfile(source,output)

    def test_offline_values_resume_preview_and_training_evaluation(self):
        torch.set_num_threads(2)
        with TemporaryDirectory() as temp:
            root=Path(temp); opts=self.options(root)
            originals=fixture.SceneDatasetTests().prepare(opts['OUTPUT_DIR'])
            hashes={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in opts['OUTPUT_DIR'].rglob('*') if p.is_file()}
            with patch.multiple(tools,**opts),patch.object(snap_preprocess,'snap_runtime',return_value={'test':'1'}),patch.object(snap_preprocess.SnapSpeckleFilter,'run',side_effect=self.fake_gpt) as gpt:
                tools.process_stage('filtered')
                record=tools.process_stage('normalized')
                self.assertTrue(record['complete'])
                self.assertLess(record['mean'][0],3)  # val=1000不参与统计。
                for split in ('train','val'):
                    with rasterio.open(opts['NORMALIZED_ROOT']/'images'/split/(split+'.tif')) as f:
                        values=f.read(1); source=originals[split][0]
                        valid=np.isfinite(source)
                        expected=(source[valid]-record['mean'][0])/record['std'][0]
                        np.testing.assert_allclose(values[valid],expected,rtol=2e-6,atol=2e-6)
                        self.assertTrue(np.isnan(values[~valid]).all())
                stamp=(opts['NORMALIZED_ROOT']/'images/train/train.tif').stat().st_mtime_ns
                tools.process_stage('filtered');tools.process_stage('normalized')
                self.assertEqual(gpt.call_count,2)
                self.assertEqual((opts['NORMALIZED_ROOT']/'images/train/train.tif').stat().st_mtime_ns,stamp)
                row=tools.export_scene(opts['OUTPUT_DIR']/'images/train/train.tif')
                self.assertEqual(row['available_stages'],['org','filtered','normalized'])
                self.assertEqual(tools.export_scene(opts['OUTPUT_DIR']/'images/train/train.tif')['status'],'reused')
                normal=opts['NORMALIZED_ROOT']
                kwargs=dict(model_class=lambda **kw:fixture.TinySceneNet(),model_kwargs={"num_classes":1},model_info=dict(model_name='tiny'),
                    image_dir=normal/'images',label_dir=normal/'labels',output_dir=root/'runs',
                    bands=(1,),batch_size=2,epochs=1,loss_name='bce',require_normalized=True,require_preprocessed=True,
                    scene_options=dict(cache_dir=root/'windows',window_size=32,halo=8,num_workers=0,windows_per_epoch=2))
                trainer=WaterSegmentationTrainer(**kwargs)
                trainer.train();trainer.dataset.close()
                saved=trainer.run_dir/'last.pth'
                checkpoint=torch.load(saved,weights_only=True,map_location='cpu')
                self.assertTrue(checkpoint['inputs_normalized'])
                self.assertEqual(checkpoint['input_preprocessing']['mean'],record['mean'])
                check_preprocessed_input(normal/'images',checkpoint)
                with self.assertRaisesRegex(ValueError,'归一化状态'):
                    check_preprocessed_input(opts['OUTPUT_DIR']/'images',checkpoint)
                # 故意设置非恒等检查点mean/std；离线输入的评价仍应跳过标准化。
                checkpoint['mean']=torch.tensor([50.]);checkpoint['std']=torch.tensor([3.])
                alt=root/'offline.pth';torch.save(checkpoint,alt)
                with patch.object(model_eval,'load_model',return_value=trainer.model),patch.object(model_eval,'SARSceneDataset',side_effect=lambda folder,cache,**kw:SARSceneDataset(folder,root/'windows',**kw)):
                    result=model_eval.evaluate(alt,use_amp=False)
                self.assertAlmostEqual(result['loss'],trainer.phase_metrics['val']['loss'],places=5)
            for path,digest in hashes.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),digest)

    def test_filter_interruption_reuses_completed_scene(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);opts=self.options(root);fixture.SceneDatasetTests().prepare(opts['OUTPUT_DIR'])
            calls=[]
            def interrupted(source,bands,output,work,diagnostics):
                calls.append(source.name)
                if len(calls)==2:raise RuntimeError('interrupted')
                self.fake_gpt(source,bands,output,work,diagnostics)
            with patch.multiple(tools,**opts),patch.object(snap_preprocess,'snap_runtime',return_value={'test':'1'}),patch.object(snap_preprocess.SnapSpeckleFilter,'run',side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'interrupted'):tools.process_stage('filtered')
                self.assertTrue((opts['FILTERED_ROOT']/'.preprocessing_incomplete').exists())
                first=opts['FILTERED_ROOT']/'images/train/train.tif';stamp=first.stat().st_mtime_ns
                tools.process_stage('filtered')
                self.assertEqual(len(calls),3)
                self.assertEqual(first.stat().st_mtime_ns,stamp)
                self.assertFalse((opts['FILTERED_ROOT']/'.preprocessing_incomplete').exists())

    def test_calibration_requires_physical_contract(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);opts=self.options(root);fixture.SceneDatasetTests().prepare(opts['OUTPUT_DIR'])
            with patch.multiple(tools,**opts),patch.object(tools,'CALIBRATION_QUANTITY',None):
                with self.assertRaisesRegex(ValueError,'校正尚未配置'):tools.process_stage('calibrated')
                self.assertFalse(opts['CALIBRATED_ROOT'].exists())

    def test_import_calibrated_products_then_normalize_and_preview(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);opts=self.options(root);fixture.SceneDatasetTests().prepare(opts['OUTPUT_DIR'])
            for split in ('train','val'):
                folder=root/'physical'/split;folder.mkdir(parents=True)
                shutil.copyfile(opts['OUTPUT_DIR']/'images'/split/(split+'.tif'),folder/(split+'.tif'))
            with patch.multiple(tools,**opts),patch.multiple(tools,CALIBRATION_QUANTITY='sigma0',CALIBRATION_REFERENCE='synthetic test only',CALIBRATION_PRODUCTS_DIR=root/'physical',CALIBRATION_COMMAND=None,NORMALIZE_INPUT_STAGE='calibrated'):
                tools.process_stage('calibrated');tools.process_stage('normalized')
                result=tools.export_scene(opts['OUTPUT_DIR']/'images/train/train.tif')
                self.assertEqual(result['available_stages'],['org','calibrated','normalized'])

    def test_preprocess_switches_and_individual_modes(self):
        with patch.multiple(tools,MODE='preprocess',ENABLE_CALIBRATION=False,ENABLE_FILTER=True,FILTER_INPUT_STAGE='org',NORMALIZE_INPUT_STAGE='filtered'),patch.object(tools,'process_stage') as run:
            tools.main()
            self.assertEqual([call.args[0] for call in run.call_args_list],['filtered','normalized'])
        with patch.multiple(tools,MODE='preprocess',ENABLE_CALIBRATION=True,ENABLE_FILTER=True,FILTER_INPUT_STAGE='calibrated',NORMALIZE_INPUT_STAGE='filtered'),patch.object(tools,'process_stage') as run:
            tools.main()
            self.assertEqual([call.args[0] for call in run.call_args_list],['calibrated','filtered','normalized'])
        for mode,stage in [('filter','filtered'),('normalize','normalized'),('calibrate','calibrated')]:
            with patch.object(tools,'MODE',mode),patch.object(tools,'process_stage') as run:
                tools.main()
                run.assert_called_once_with(stage)

    def test_external_calibration_command_and_wsl_placeholders(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);opts=self.options(root);fixture.SceneDatasetTests().prepare(opts['OUTPUT_DIR'])
            commands=[]
            def external(command,**kwargs):
                commands.append(command)
                self.assertTrue(command[3].startswith('/mnt/'))
                self.assertNotIn(chr(92),command[3])
                shutil.copyfile(command[1],command[2])
            with patch.multiple(tools,**opts),patch.multiple(tools,CALIBRATION_QUANTITY='gamma0',CALIBRATION_REFERENCE='synthetic mock command',CALIBRATION_PRODUCTS_DIR=None,CALIBRATION_COMMAND=['mock_gamma','{input}','{output}','{input_wsl}']),patch.object(tools.subprocess,'run',side_effect=external):
                tools.process_stage('calibrated')
                tools.process_stage('calibrated')
                self.assertEqual(len(commands),2)
