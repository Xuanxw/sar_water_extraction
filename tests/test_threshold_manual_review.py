"""人工核验：验证分层加权、缺失参考阻断、分景进度和点位保护。"""
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import fiona
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_threshold_review import weighted_metrics, evaluate, prepare


class ManualReviewTests(unittest.TestCase):
    def points(self):
        return [dict(point_id=str(i), scene_id='scene', stratum='shore_buffer' if i < 2 else 'nonwater_other',
                     pred=1 if i < 2 else 0, population=100 if i < 2 else 900,
                     sample_n=2, pixel_m2=2 if i < 2 else 1) for i in range(4)]

    def fixture(self, root, refs):
        frame = {'scope': 'scene_observations_not_unique_geographic_area', 'points': self.points(),
                 'scenes': [{'scene_id': 'scene', 'date': '2025-01-01',
                             'strata': {'shore_buffer': {}, 'nonwater_other': {}}}]}
        (root / 'sample_frame.json').write_text(json.dumps(frame), encoding='utf-8')
        schema = {'geometry': 'Point', 'properties': {'point_id': 'str', 'ref_class': 'int'}}
        with fiona.open(root / 'manual_review.gpkg', 'w', driver='GPKG', crs='EPSG:4326',
                        layer='verification_points', schema=schema) as dst:
            for key, ref in refs.items():
                dst.write({'geometry': {'type': 'Point', 'coordinates': (116, 40)},
                           'properties': {'point_id': key, 'ref_class': ref}})

    def test_weighted_counts_not_naive_sample_ratio(self):
        refs = {'0': 1, '1': 0, '2': 1, '3': 0}
        values = weighted_metrics(self.points(), refs)
        self.assertEqual([values[k] for k in ('TN', 'FP', 'FN', 'TP')], [450, 50, 450, 50])
        self.assertAlmostEqual(values['Precision'], .5)
        self.assertAlmostEqual(values['Recall'], .1)
        self.assertAlmostEqual(values['F1'], 1 / 6)
        areas = weighted_metrics(self.points(), refs, area_weight=True)
        self.assertEqual(areas['TP'], 100)
        self.assertAlmostEqual(areas['Recall'], 100 / 550)

    def test_incomplete_or_uncertain_does_not_become_nonwater(self):
        for value in (None, -1):
            with self.subTest(value=value), TemporaryDirectory() as temp:
                root = Path(temp)
                self.fixture(root, {'0': 1, '1': 0, '2': 1, '3': value})
                report = evaluate(root)
                self.assertEqual(report['status'], 'awaiting_reference')
                self.assertEqual(report['overall'], [])
                self.assertEqual(report['reviewed'], 3)
                before = (root / 'manual_review.gpkg').read_bytes()
                prepare(root / 'nonexistent_results', root)
                self.assertEqual(before, (root / 'manual_review.gpkg').read_bytes())

    def test_complete_scene_and_boundary_report(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root, {'0': 1, '1': 0, '2': 1, '3': 0})
            report = evaluate(root)
            self.assertEqual(report['status'], 'estimated')
            self.assertEqual(len(report['overall']), 4)
            self.assertAlmostEqual(report['overall'][0]['Recall'], .1)
            self.assertEqual(report['overall'][2]['Recall'], 1)

    def test_deleted_points_rejected(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root, {'0': 1, '1': 0, '2': 1})
            with self.assertRaisesRegex(ValueError, '删除'):
                evaluate(root)


if __name__ == '__main__':
    unittest.main()
