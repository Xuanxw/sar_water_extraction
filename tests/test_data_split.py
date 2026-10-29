"""内容分组不能跨侧；搬迁保持字节、可重跑、可恢复，异常必须先于任何移动。"""
import csv
import hashlib
import sys
import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import rasterio

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"),
                str(Path(__file__).resolve().parents[1] / "scripts")]
from prepare_train_val_split import assign_split, file_hash, move_split, scan_samples, write_csv


class DataSplitTests(unittest.TestCase):
    def make_data(self, folder):
        """构造重复、异标签、同辅助层和稀有场景；测试只操作临时目录。"""
        root = Path(folder)
        images, labels = root / "images", root / "labels"
        images.mkdir(); labels.mkdir()
        catalog = []
        for index in range(12):
            name = f"{index:02}.tif"
            data = np.zeros((6, 12, 12), dtype=np.int16)
            data[0] = index + 1
            data[1] = index + 100
            data[2] = np.arange(144).reshape(12, 12) + index * 1000
            data[3] = data[2] + 1
            data[4] = 40
            data[5] = index
            label = np.zeros((12, 12), dtype=np.uint8)
            label[4:8, 4:8] = 1
            if index in (1, 3):
                # 00/01完全重复；02/03输入相同但03标签不同。
                with rasterio.open(images / f"{index-1:02}.tif") as source:
                    data = source.read()
                if index == 3:
                    label[3, 3] = 1
            if index == 5:
                with rasterio.open(images / "04.tif") as source:
                    data[2:6] = source.read((3, 4, 5, 6))
            if index in (6, 7):
                data[2:6] = 0  # 平坦辅助层一致不得误当同范围线索。
            with rasterio.open(images / name, "w", driver="GTiff", height=12, width=12,
                               count=6, dtype="int16") as target:
                target.write(data)
            label_path = labels / f"{index:02}.png"
            with rasterio.open(label_path, "w", driver="PNG", height=12, width=12,
                               count=1, dtype="uint8") as target:
                target.write(label, 1)
            catalog.append(dict(name=name, water_context_scene="建成区" if index == 11 else "农田",
                                label_sha256=file_hash(label_path)))
        scene_csv = root / "scenes.csv"
        write_csv(scene_csv, catalog)
        return root, scene_csv

    def setUp(self):
        warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

    def test_duplicates_conflicts_auxiliary_and_rare_scene(self):
        with TemporaryDirectory() as folder:
            root, scene_csv = self.make_data(folder)
            samples = scan_samples(root, scene_csv)
            rows = assign_split(samples)
            again = assign_split(list(reversed(samples)))
            self.assertEqual(rows, again)
            records = {r["name"]: r for r in rows}
            self.assertEqual(records["00.tif"]["active"], 1)
            self.assertEqual(records["01.tif"]["reason"], "duplicate")
            for name in ("02.tif", "03.tif"):
                self.assertEqual(records[name]["reason"], "label_conflict")
                self.assertEqual(records[name]["active"], 0)
            self.assertEqual(records["04.tif"]["group_id"], records["05.tif"]["group_id"])
            self.assertTrue(all(records[n]["active"] for n in ("04.tif", "05.tif")))
            self.assertNotEqual(records["06.tif"]["group_id"], records["07.tif"]["group_id"])
            self.assertEqual(records["11.tif"]["split"], "train")
            for group_id in {r["group_id"] for r in rows}:
                self.assertEqual(len({r["split"] for r in rows if r["group_id"] == group_id}), 1)
            self.assertEqual(len(rows), 12)
            self.assertEqual(sum(r["active"] for r in rows), 9)
            self.assertGreater(sum(r["active"] for r in rows if r["split"] == "val"), 0)
            # 场景标签文件哈希过期时不沿用旧标签场景。
            scene_text = scene_csv.read_text(encoding="utf-8-sig")
            scene_csv.write_text(scene_text.replace(file_hash(root / "labels/11.png"), "stale"), encoding="utf-8-sig")
            changed = {r["name"]: r for r in scan_samples(root, scene_csv)}
            self.assertEqual(changed["11.tif"]["scene"], "unknown")

    def test_move_retry_and_restore_preserve_every_file(self):
        with TemporaryDirectory() as folder:
            root, scene_csv = self.make_data(folder)
            manifest = root / "plan/split_manifest.csv"
            rows = assign_split(scan_samples(root, scene_csv))
            write_csv(manifest, rows)
            before = {p.relative_to(root).as_posix(): file_hash(p)
                      for directory in (root / "images", root / "labels") for p in directory.iterdir()}
            move_split(manifest, root)
            move_split(manifest, root)  # 原计划重复执行不再次划分。
            self.assertEqual(len(list((root / "images").rglob("*.tif"))), 12)
            self.assertEqual(len(list((root / "labels").rglob("*.png"))), 12)
            self.assertTrue((root / "split_manifest.csv").exists())
            for row in rows:
                self.assertEqual(file_hash(root / "images" / row["image_relative"]), row["image_file_sha256"])
                self.assertEqual(file_hash(root / "labels" / row["label_relative"]), row["label_file_sha256"])
            move_split(manifest, root, restore=True)
            after = {p.relative_to(root).as_posix(): file_hash(p)
                     for directory in (root / "images", root / "labels") for p in directory.iterdir()}
            self.assertEqual(before, after)
            self.assertTrue(manifest.exists())
            self.assertFalse((root / "split_manifest.csv").exists())
            self.assertTrue((manifest.parent / "move_log.csv").exists())
            self.assertTrue((manifest.parent / "restore_log.csv").exists())

    def test_preflight_rejects_conflict_and_path_escape_before_moving(self):
        with TemporaryDirectory() as folder:
            root, scene_csv = self.make_data(folder)
            rows = assign_split(scan_samples(root, scene_csv))
            manifest = root / "plan/split_manifest.csv"
            write_csv(manifest, rows)
            blocked = root / "images" / rows[-1]["image_relative"]
            blocked.parent.mkdir(parents=True)
            blocked.write_bytes(b"existing file")
            with self.assertRaises(FileExistsError):
                move_split(manifest, root)
            self.assertEqual(len(list((root / "images").glob("*.tif"))), 12)
            self.assertEqual(blocked.read_bytes(), b"existing file")
            blocked.unlink()
            rows[-1]["image_relative"] = "../outside.tif"
            write_csv(manifest, rows)
            with self.assertRaises(ValueError):
                move_split(manifest, root)
            self.assertEqual(len(list((root / "images").glob("*.tif"))), 12)
            self.assertFalse((manifest.parent / "move_log.csv").exists())


if __name__ == "__main__":
    unittest.main()
