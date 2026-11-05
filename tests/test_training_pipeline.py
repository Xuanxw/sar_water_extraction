"""验证训练与后处理衔接，防止误读旧实验或在训练失败后继续处理。"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"),
                str(Path(__file__).resolve().parents[1] / "scripts")]
from supervised import train
from supervised.evaluation import postprocess


class TrainingPipelineTests(unittest.TestCase):
    def test_uses_current_best_checkpoint_and_saved_validation_after_training(self):
        for binary, filename in ((True, "best_f1.pth"), (False, "best.pth")):
            events = []
            trainer = SimpleNamespace(
                train=lambda: events.append("train"), run_dir=Path("current_run"),
                criterion=SimpleNamespace(binary=binary), image_dir=Path("current_images"),
                label_dir=Path("current_labels"), batch_size=2, use_amp=False)
            def process(paths, **options):
                self.assertEqual(events, ["train"])
                self.assertEqual(paths, [Path("current_run") / filename])
                self.assertEqual(options["image_dir"], Path("current_images"))
                self.assertTrue(options["use_saved_val"] and options["with_labels"])
                self.assertIsNone(options["output_dir"])
                return {"done": True}
            with patch.object(train, "DATA_LAYOUT", "patch"), patch.object(train, "RUN_MODE", "train_postprocess"), \
                 patch.object(train, "POSTPROCESS_CHECKPOINTS", ["old_run/best.pth"]), \
                 patch.object(train, "create_trainer", return_value=trainer), \
                 patch.object(postprocess, "run_postprocess", side_effect=process), \
                 patch.object(train.torch.cuda, "is_available", return_value=False):
                self.assertEqual(train.main(), {"done": True})

    def test_postprocess_only_never_starts_training(self):
        options = dict(use_saved_val=False, with_labels=False, image_dir="test_images")
        with patch.object(train, "DATA_LAYOUT", "patch"), patch.object(train, "RUN_MODE", "postprocess"), \
             patch.object(train, "POSTPROCESS_CHECKPOINTS", ["existing/best_f1.pth"]), \
             patch.object(train, "POSTPROCESS_OPTIONS", options), \
             patch.object(train, "create_trainer") as create, \
             patch.object(postprocess, "run_postprocess") as process:
            train.main()
            create.assert_not_called()
            process.assert_called_once_with(["existing/best_f1.pth"], **options)

    def test_training_only_and_training_failure_do_not_start_postprocess(self):
        trainer = Mock(run_dir=Path("run"))
        with patch.object(train, "create_trainer", return_value=trainer), \
             patch.object(postprocess, "run_postprocess") as process:
            with patch.object(train, "DATA_LAYOUT", "patch"), patch.object(train, "RUN_MODE", "train"):
                self.assertEqual(train.main(), Path("run"))
            trainer.train.side_effect = RuntimeError("training failed")
            with patch.object(train, "DATA_LAYOUT", "patch"), patch.object(train, "RUN_MODE", "train_postprocess"), self.assertRaises(RuntimeError):
                train.main()
            process.assert_not_called()

    def test_missing_postprocess_paths_do_not_fall_back_to_old_experiment(self):
        with patch.object(train, "DATA_LAYOUT", "patch"), patch.object(train, "RUN_MODE", "postprocess"), \
             patch.object(train, "POSTPROCESS_CHECKPOINTS", []), \
             patch.object(train, "create_trainer") as create, self.assertRaises(ValueError):
            train.main()
        create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
