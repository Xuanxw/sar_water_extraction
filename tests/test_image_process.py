"""离线滤波、归一化、快视图、训练输入及续跑，只用合成数据与模拟GPT。"""
import json
import csv
import hashlib
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, MagicMock
import numpy as np
import rasterio
import torch
from datasets import image_process as tools, snap_preprocess
from supervised.load.sar_scene_dataset import SARSceneDataset, check_preprocessed_input
from supervised.train import WaterSegmentationTrainer
from supervised.evaluation import model_eval
import test_sar_scene_dataset as fixture

class ImageProcessTests(unittest.TestCase):
    def setUp(self):
        cpu = patch.object(torch.cuda,'is_available',return_value=False)
        cpu.start();self.addCleanup(cpu.stop)

    def test_all_orders_stages_and_stops_if_prepare_incomplete(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);calls=[]
            def prepare(**kwargs):
                self.assertEqual(kwargs,dict(mode='prepare',reuse_complete=True))
                calls.append('prepare');(root/'split_manifest.csv').write_text('ready')
            with patch.multiple(tools,MODE='all',OUTPUT_DIR=root,ENABLE_CALIBRATION=False,ENABLE_FILTER=True,
                    FILTER_INPUT_STAGE='org',NORMALIZE_INPUT_STAGE='filtered'), \
                    patch.object(tools,'prepare_dataset',side_effect=prepare), \
                    patch.object(tools,'process_stage',side_effect=lambda s:calls.append(s)), \
                    patch.object(tools,'export_quicklooks',side_effect=lambda:calls.append('quicklook')):
                tools.main()
                self.assertEqual(calls,['prepare','filtered','normalized','quicklook'])
                calls.clear();(root/'split_manifest.csv').unlink()
                with patch.object(tools,'prepare_dataset',return_value=None),self.assertRaisesRegex(RuntimeError,'尚未全部完成'):
                    tools.main()
                self.assertEqual(calls,[])

    def test_completed_prepare_reuses_files_without_rebuilding(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);output=root/'prepared';table=root/'split.csv'
            rows=[dict(影像编号=f'BC3-SM-ORG-VV-20240101T000000-00000{i}',数据划分=split,
                       场景=f'省；G_abcdef00000{i}',是否采用='已确认',准备批次='bat2')
                  for i,split in enumerate(('train','val'),1)]
            with table.open('w',encoding='utf-8-sig',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
            vector=MagicMock();vector.__enter__.return_value.__iter__.side_effect=lambda:iter([{'properties':{}}])
            def prepare(row,*args,**kwargs):
                record=dict.fromkeys(tools.MANIFEST_FIELDS,'')
                record.update(name=row['scene_id']+'.tif',label_name=row['scene_id']+'.tif',scene_id=row['scene_id'],
                              split=row['split'],active=1,water_pixels=1,background_pixels=1)
                for kind in ('images','labels'):
                    path=output/kind/row['split']/record['name'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'test')
                return [record]
            with patch.multiple(tools,OUTPUT_DIR=output,SPLIT_CSV=table,PREPARE_BATCH='bat2',MAX_SCENES=None,
                    CLASS_FIELD=None,CLASS_MAP={None:1},SCENE_CLASS_RULES={},VALID_MASK_DIR=None), \
                    patch.object(tools,'index_sources',return_value=({},{})), \
                    patch.object(tools,'select_sources',return_value=('image','label')), \
                    patch.object(tools.fiona,'open',return_value=vector),patch.object(tools,'prepare_scene',side_effect=prepare) as build:
                tools.prepare_dataset(mode='prepare',reuse_complete=True)
                manifest=output/'split_manifest.csv';before=manifest.read_bytes();stamp=manifest.stat().st_mtime_ns
                tools.prepare_dataset(mode='prepare',reuse_complete=True)
                self.assertEqual(build.call_count,2)
                self.assertEqual(manifest.read_bytes(),before);self.assertEqual(manifest.stat().st_mtime_ns,stamp)

    def test_reference_statistics_skip_reestimation_and_resume(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);opts=self.options(root)
            fixture.SceneDatasetTests().prepare(opts['OUTPUT_DIR'])
            with patch.multiple(tools,**opts),patch.object(snap_preprocess,'snap_runtime',return_value={'test':'1'}), \
                    patch.object(snap_preprocess.SnapSpeckleFilter,'run',side_effect=self.fake_gpt):
                tools.process_stage('filtered');first=tools.process_stage('normalized')
                reference=opts['NORMALIZED_ROOT']/'preprocessing.json'
                target=root/'batch2_normalized'
                with patch.multiple(tools,NORMALIZATION_REFERENCE=reference,NORMALIZED_ROOT=target), \
                        patch.object(SARSceneDataset,'normalization',side_effect=AssertionError('不能重新估计统计')):
                    second=tools.process_stage('normalized')
                    self.assertEqual(first['mean'],second['mean']);self.assertEqual(first['std'],second['std'])
                    for split in ('train','val'):
                        with rasterio.open(opts['NORMALIZED_ROOT']/f'images/{split}/{split}.tif') as a,rasterio.open(target/f'images/{split}/{split}.tif') as b:
                            np.testing.assert_array_equal(a.read(),b.read())
                    stamp=(target/'images/train/train.tif').stat().st_mtime_ns
                    tools.process_stage('normalized')
                    self.assertEqual((target/'images/train/train.tif').stat().st_mtime_ns,stamp)
                    broken=dict(first,complete=False);reference.write_text(json.dumps(broken),encoding='utf-8')
                    with self.assertRaisesRegex(ValueError,'尚未完整完成'):tools.process_stage('normalized')

    def options(self, root):
        return dict(OUTPUT_DIR=root/'raw',CALIBRATED_ROOT=root/'processed/calibrated',
                    FILTERED_ROOT=root/'processed/filtered',NORMALIZED_ROOT=root/'processed/normalized',
                    WINDOW_CACHE_DIR=root/'windows',SNAP_CACHE_DIR=root/'snap',SNAP_WORK_DIR=root/'ssd_work',NORMALIZE_WORK_DIR=root/'normalize_work',
                    WINDOW_SIZE=32,HALO=8,NORMALIZATION_WINDOWS=2,SEED=42,BANDS=(1,),
                    NORMALIZATION_REFERENCE=None,
                    CLEAN_SNAP_CACHE_ON_SUCCESS=False,ENABLE_CALIBRATION=False,ENABLE_FILTER=True,FILTER_INPUT_STAGE='org',NORMALIZE_INPUT_STAGE='filtered')

    def fake_gpt(self,source,bands,output,work,diagnostics):
        self.assertTrue(str(work).startswith(str(snap_preprocess.WORK_DIR)))
        self.assertNotEqual(work.parent,diagnostics)
        shutil.copyfile(source,output)

    def test_offline_values_resume_preview_and_training_evaluation(self):
        torch.set_num_threads(2)
        with TemporaryDirectory() as temp:
            root=Path(temp); opts=self.options(root)
            originals=fixture.SceneDatasetTests().prepare(opts['OUTPUT_DIR'])
            hashes={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in opts['OUTPUT_DIR'].rglob('*') if p.is_file()}
            with patch.multiple(tools,**opts),patch.object(snap_preprocess,'snap_runtime',return_value={'test':'1'}),patch.object(snap_preprocess.SnapSpeckleFilter,'run',side_effect=self.fake_gpt) as gpt:
                tools.process_stage('filtered')
                for cached in (root/'snap').glob('*/filtered.tif'):
                    split=cached.parent.name.split('_')[0]
                    result=opts['FILTERED_ROOT']/'images'/split/(split+'.tif')
                    self.assertEqual(hashlib.sha256(cached.read_bytes()).digest(),hashlib.sha256(result.read_bytes()).digest())
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
            opts['CLEAN_SNAP_CACHE_ON_SUCCESS']=True
            calls=[]
            def interrupted(source,bands,output,work,diagnostics):
                calls.append(source.name)
                if len(calls)==2:raise RuntimeError('interrupted')
                self.fake_gpt(source,bands,output,work,diagnostics)
            with patch.multiple(tools,**opts),patch.object(snap_preprocess,'snap_runtime',return_value={'test':'1'}),patch.object(snap_preprocess.SnapSpeckleFilter,'run',side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'interrupted'):tools.process_stage('filtered')
                self.assertTrue((opts['FILTERED_ROOT']/'.preprocessing_incomplete').exists())
                self.assertEqual(len(list((root/'snap').glob('*/filtered.tif'))),1)
                first=opts['FILTERED_ROOT']/'images/train/train.tif';stamp=first.stat().st_mtime_ns
                tools.process_stage('filtered')
                self.assertEqual(len(calls),3)
                self.assertEqual(list((root/'snap').glob('*/filtered.tif')),[])
                self.assertTrue((opts['FILTERED_ROOT']/'cache_cleanup.json').exists())
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

    def test_existing_cache_survives_work_directory_change(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);opts=self.options(root);fixture.SceneDatasetTests().prepare(opts['OUTPUT_DIR'])
            with patch.multiple(tools,**opts),patch.object(snap_preprocess,'snap_runtime',return_value={'test':'1'}),patch.object(snap_preprocess.SnapSpeckleFilter,'run',side_effect=self.fake_gpt) as gpt:
                tools.process_stage('filtered')
                output=opts['FILTERED_ROOT']/'images/train/train.tif'
                expected=hashlib.sha256(output.read_bytes()).hexdigest()
                output.unlink()  # 仅删除本测试临时成果，模拟缓存已完成但成果尚未发布。
                new_work=root/'another_ssd_work'
                with patch.object(tools,'SNAP_WORK_DIR',new_work):
                    tools.process_stage('filtered')
                self.assertEqual(gpt.call_count,2)
                self.assertEqual(hashlib.sha256(output.read_bytes()).hexdigest(),expected)
                self.assertFalse(new_work.exists())  # 命中旧缓存，无须启动新的SSD工作任务。

    def test_fast_normalization_matches_old_block_formula_exactly(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);source=root/'input.tif'
            values=np.random.default_rng(42).normal(15,7,(2,513,777)).astype('float32')
            values[0,0,0]=0.;values[1,12,19]=np.nan;values[0,512,776]=np.inf
            transform=rasterio.transform.from_origin(100,200,3,3)
            with rasterio.open(source,'w',driver='GTiff',count=2,width=777,height=513,dtype='float32',nodata=np.nan,crs='EPSG:32650',transform=transform) as dst:
                dst.write(values)
            mean=np.array([12.123456789,16.987654321]);std=np.array([2.123456789,7.987654321])
            reference=root/'old.tif'
            with rasterio.open(source) as src:
                profile=src.profile.copy();profile.update(tiled=True,blockxsize=256,blockysize=256,compress='deflate',predictor=3)
                with rasterio.open(reference,'w',**profile) as dst:
                    for _,win in dst.block_windows(1):
                        raw=src.read((2,1),window=win,masked=True,out_dtype='float32')
                        valid=~np.ma.getmaskarray(raw).any(axis=0)&np.isfinite(raw.data).all(axis=0)
                        normalized=(raw.data-mean[:,None,None])/std[:,None,None]
                        dst.write(np.where(valid[None],normalized,np.nan).astype('float32'),window=win)
            with patch.multiple(tools,NORMALIZE_WORK_DIR=root/'ssd',NORMALIZE_READ_SIZE=1024,NORMALIZE_THREADS=2,NORMALIZE_COMPRESSION_LEVEL=1):
                timing=tools.write_normalized_scene(source,root/'fast.tif',(2,1),mean,std)
                self.assertIn('publish_copy_seconds',timing)
                self.assertFalse(list((root/'ssd').iterdir()))
            with rasterio.open(reference) as old,rasterio.open(root/'fast.tif') as fast:
                np.testing.assert_array_equal(old.read(),fast.read())
                np.testing.assert_array_equal(old.read_masks(),fast.read_masks())
                self.assertEqual(old.crs,fast.crs);self.assertEqual(old.transform,fast.transform)
                self.assertEqual(fast.read(2)[0,0],np.float32((0-mean[1])/std[1]))

    def test_success_cleans_current_batch_only_and_can_normalize(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);opts=self.options(root);opts['CLEAN_SNAP_CACHE_ON_SUCCESS']=True
            fixture.SceneDatasetTests().prepare(opts['OUTPUT_DIR'])
            unrelated=root/'snap'/'another_batch';unrelated.mkdir(parents=True)
            (unrelated/'filtered.tif').write_bytes(b'keep unrelated task')
            with patch.multiple(tools,**opts),patch.object(snap_preprocess,'snap_runtime',return_value={'test':'1'}),patch.object(snap_preprocess.SnapSpeckleFilter,'run',side_effect=self.fake_gpt) as gpt:
                result=tools.process_stage('filtered')
                self.assertTrue(result['complete'])
                self.assertEqual(list((root/'snap').glob('*/filtered.tif')),[unrelated/'filtered.tif'])
                self.assertTrue(tools.process_stage('normalized')['complete'])
                tools.process_stage('filtered')
                self.assertEqual(gpt.call_count,2)  # 正式成果存在，无缓存也不重跑GPT。
                self.assertEqual((unrelated/'filtered.tif').read_bytes(),b'keep unrelated task')
