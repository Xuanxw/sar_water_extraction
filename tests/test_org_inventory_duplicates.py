"""输入去重验证：ZIP优先但不隐式选择多个归档版本。"""
import shutil
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import test_org_pipeline as fixtures
from thres_segmentation.org_io import inventory


class InventoryDuplicateTests(unittest.TestCase):
    def test_zip_and_extracted_tiff_keep_original_source(self):
        with TemporaryDirectory() as temp:
            config = fixtures.OrgProductionTests().make_config(Path(temp))
            root = config['paths']['ORG_ROOT']
            before = inventory(root, config['org'])
            archive = next(p for p in root.glob('*ORG*.zip'))
            with ZipFile(archive) as source:
                source.extractall(root/'extracted')
            self.assertEqual(inventory(root, config['org']), before)
            self.assertTrue(any((root/'extracted').glob('*.tif')))

    def test_multiple_archives_still_need_selection(self):
        with TemporaryDirectory() as temp:
            config = fixtures.OrgProductionTests().make_config(Path(temp))
            root = config['paths']['ORG_ROOT']
            archive = next(root.glob('*ORG*.zip'))
            duplicate = root/'another'; duplicate.mkdir()
            shutil.copy2(archive, duplicate/archive.name)
            with self.assertRaisesRegex(ValueError, '同景存在多个版本'):
                inventory(root, config['org'])

    def test_multiple_standalone_tiffs_still_need_selection(self):
        with TemporaryDirectory() as temp:
            config = fixtures.OrgProductionTests().make_config(Path(temp))
            root = config['paths']['ORG_ROOT']
            tif = next(root.glob('*ORG*.tif'))
            duplicate = root/'another'; duplicate.mkdir()
            shutil.copy2(tif, duplicate/tif.name)
            with self.assertRaisesRegex(ValueError, '同景存在多个版本'):
                inventory(root, config['org'])


if __name__ == '__main__':
    unittest.main()
