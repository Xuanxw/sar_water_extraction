"""Exercise the four DFC terrain YAMLs through the training entry point on small CPU fixtures."""
import csv
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from datasets.dfc_terrain import DFC_TERRAIN_INPUTS, dfc_terrain_from_checkpoint
from supervised import train
from supervised.models.checkpoint import load_model


class DfcTerrainTrainingTests(unittest.TestCase):
    def test_same_model_initialization_and_checkpoint_input_contract(self):
        with TemporaryDirectory() as tmp, patch.dict(vars(train)), \
                patch.object(torch.cuda, "is_available", return_value=False):
            root = Path(tmp)
            rows = []
            row, col = np.indices((32, 32), dtype=np.float32)
            grid = dict(driver="GTiff", width=32, height=32, crs="EPSG:32650",
                        transform=from_origin(500000, 4000000, 10, 10))
            for number, split in enumerate(("train", "train", "val")):
                name = f"{number}.tif"
                images, labels = [root / folder / split for folder in ("images", "labels")]
                images.mkdir(parents=True, exist_ok=True)
                labels.mkdir(parents=True, exist_ok=True)
                with rasterio.open(images / name, "w", **grid, count=4, dtype="float32") as dst:
                    dst.write(np.stack((1 + row + col + number, col, row, col**2 - row)))
                with rasterio.open(labels / name, "w", **grid, count=1, dtype="uint8") as dst:
                    dst.write((row < col).astype("uint8"), 1)
                rows.append(dict(name=name, split=split, active=1, group_id=number))
            with (root / "split_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)

            def make(mode, resume=None):
                filename = (f"train_pvt_deeplabv3plus_log_terrain_{mode}.yaml" if mode else
                            "train_pvt_deeplabv3plus_log_trainnorm80.yaml")
                train.configure(ROOT / "configs/s1" / filename)
                with patch.multiple(train, IMAGE_DIR=root / "images", LABEL_DIR=root / "labels",
                                    OUTPUT_DIR=root / "runs" / (mode or "original"),
                                    RESUME_CHECKPOINT=resume):
                    return train.create_trainer()

            digests, vv_and_masks = [], []
            for mode in DFC_TERRAIN_INPUTS:
                with self.subTest(mode=mode):
                    trainer = make(mode)
                    self.assertEqual(trainer.bands, (1, 4))
                    self.assertEqual(trainer.in_channels, 3)
                    sample = trainer.val_set[0]
                    image = (sample["image"].unsqueeze(0) - trainer.mean) / trainer.std
                    self.assertTrue(torch.isfinite(image).all())
                    for slot, kinds in ((1, ("vv_only", "slope")), (2, ("vv_only", "elevation"))):
                        if mode in kinds:
                            self.assertEqual(torch.count_nonzero(image[0, slot]).item(), 0)
                    vv_and_masks.append((image[0, 0], sample["valid_mask"]))
                    digest = hashlib.sha256()
                    for tensor in trainer.model.state_dict().values():
                        digest.update(tensor.detach().numpy().tobytes())
                    digests.append(digest.hexdigest())
                    if mode == "vv_only":
                        trainer.model.eval()
                        with torch.inference_mode():
                            expected = trainer.model(image)
                        trainer.phase_metrics = {}  # Snapshot initialized weights without updating the model.
                        trainer.recent_train_files = []
                        trainer.save_checkpoint(1, select_best=False)
                        saved_path = trainer.run_dir / "last.pth"
                        saved = torch.load(saved_path, weights_only=True, map_location="cpu", mmap=True)
                        self.assertEqual(dfc_terrain_from_checkpoint(saved), mode)
                        self.assertEqual(json.loads((trainer.run_dir / "dfc_terrain.json").read_text("utf-8")),
                                         saved["dfc_terrain"])
                        with torch.inference_mode():
                            torch.testing.assert_close(load_model(saved).eval()(image), expected, rtol=0, atol=0)
                        del saved
                    del trainer
            self.assertEqual(len(set(digests)), 1)
            for vv, mask in vv_and_masks[1:]:
                torch.testing.assert_close(vv, vv_and_masks[0][0], rtol=0, atol=0)
                torch.testing.assert_close(mask, vv_and_masks[0][1])
            resumed = make("vv_only", saved_path)
            self.assertEqual(resumed.start_epoch, 2)
            del resumed
            for other in ("slope", None):
                with self.assertRaisesRegex(ValueError, "地形"):
                    make(other, saved_path)
            legacy_path = root / "legacy.pth"
            torch.save({"bands": (1, 4), "input_channels": ("VV", "DEM")}, legacy_path)
            with self.assertRaisesRegex(ValueError, "地形"):
                make("vv_only", legacy_path)


if __name__ == "__main__":
    unittest.main()
