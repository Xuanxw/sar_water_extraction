"""场景规则不把无水图、未知覆盖或永久水体误分为农田等陆地背景。"""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"),
                str(Path(__file__).resolve().parents[1] / "scripts")]
from classify_scenes import classify_arrays, describe_region


class SceneClassificationTests(unittest.TestCase):
    def test_land_context_excludes_water_but_retains_counts(self):
        cover = np.full((20, 20), 80)
        cover[:5] = 40
        region = np.ones_like(cover, dtype=bool)
        tile = describe_region(cover, region, region)
        land = describe_region(cover, region, region, land_only=True)
        self.assertEqual(tile['primary'], '永久水体')
        self.assertEqual(land['primary'], '农田')
        self.assertEqual(land['counts'][80], 300)
        self.assertEqual(land['shares'][40], 1)

    def test_unknown_and_sparse_land_stay_unclassified(self):
        cover = np.full((20, 20), 80)
        region = np.ones_like(cover, dtype=bool)
        valid = region.copy()
        valid[:10] = False
        self.assertEqual(describe_region(cover, valid, region)['primary'], '辅助数据不足')
        self.assertEqual(describe_region(cover, region, region, land_only=True)['primary'], '陆地背景不足')

    def test_no_water_has_no_artificial_buffer(self):
        label = np.zeros((40, 40), dtype=np.uint8)
        valid = np.ones_like(label, dtype=bool)
        tile, inside, context = classify_arrays(np.full_like(label, 50), valid, label, valid)
        self.assertEqual(tile['primary'], '建成区')
        self.assertEqual(inside['total'], 0)
        self.assertEqual(context['total'], 0)
        self.assertEqual(context['primary'], '无标签水体')

    def test_euclidean_ring_and_multiple_backgrounds(self):
        label = np.zeros((40, 40), dtype=np.uint8)
        label[15:25, 15:25] = 1
        valid = np.ones_like(label, dtype=bool)
        cover = np.full_like(label, 40)
        cover[:, 20:] = 50
        tile, inside, context = classify_arrays(cover, valid, label, valid)
        self.assertEqual(inside['total'], 100)
        self.assertEqual(set(context['tags']), {'农田', '建成区'})
        # 距离角点超过5的(10,10)不应计入欧氏环带，区别于正方形膨胀。
        cover[10, 10] = 10
        changed = classify_arrays(cover, valid, label, valid)[2]
        self.assertEqual(changed['counts'][10], 0)
        valid[14, 20] = False
        changed = classify_arrays(cover, valid, label, valid)[2]
        self.assertEqual(changed['total'], context['total'] - 1)


if __name__ == '__main__':
    unittest.main()
