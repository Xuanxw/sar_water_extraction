"""训练和后处理入口分离：禁止隐式联动或回退旧模型。"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from supervised import train, post_process
from supervised.evaluation import postprocess

# 测试显式加载样例配置，正式脚本仅使用命令行提供的文件。
train.configure(Path(__file__).resolve().parents[1] / 'configs/train.yaml')
post_process.configure(Path(__file__).resolve().parents[1] / 'configs/post_process.yaml')


class TrainingPipelineTests(unittest.TestCase):
    def test_training_success_and_failure_never_start_postprocess(self):
        with TemporaryDirectory() as temp:
            trainer=Mock(run_dir=Path(temp))
            with patch.object(train,'create_trainer',return_value=trainer), patch.object(postprocess,'run_postprocess') as process:
                self.assertEqual(train.main(),Path(temp))
                trainer.train.side_effect=RuntimeError('training failed')
                with self.assertRaises(RuntimeError):train.main()
                process.assert_not_called()

    def test_patch_postprocess_only_never_starts_training(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);checkpoint=root/'model.pth';checkpoint.touch()
            paths=dict(post_process.PATHS,POSTPROCESS_CHECKPOINTS=[checkpoint],OUTPUT_DIR=root/'post')
            config=dict(post_process.CONFIG,run=dict(DATA_LAYOUT='patch'))
            with patch.object(post_process,'PATHS',paths),patch.object(post_process,'CONFIG',config), \
                 patch.object(train,'create_trainer') as create,patch.object(postprocess,'run_postprocess') as process:
                post_process.main()
            create.assert_not_called()
            self.assertEqual(process.call_args.args[0],[checkpoint])
            self.assertEqual(process.call_args.kwargs['output_dir'],root/'post/quality')
            self.assertTrue(process.call_args.kwargs['save_visuals'])

    def test_missing_postprocess_paths_do_not_fall_back(self):
        with patch.object(post_process,'PATHS',dict(post_process.PATHS,POSTPROCESS_CHECKPOINTS=[])), \
             patch.object(train,'create_trainer') as create,self.assertRaises(FileNotFoundError):
            post_process.main()
        create.assert_not_called()
