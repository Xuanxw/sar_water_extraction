"""正式入口回归：最终快视图使用清理后栅格，前置文件归档，续跑不重算。"""
import hashlib,importlib.util,json,sys,unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import yaml
import test_org_pipeline as fixtures
from thres_segmentation.main import main
from thres_segmentation import org_pipeline as pipeline

class MainProductsTests(unittest.TestCase):
    def test_main_outputs_and_archive_resume(self):
        with TemporaryDirectory() as d:
            root=Path(d);config=fixtures.OrgProductionTests().make_config(root)
            config['org']['archive_intermediates']=True
            path=root/'config.yaml';path.write_text(yaml.safe_dump(json.loads(json.dumps(config,default=str)),allow_unicode=True),encoding='utf-8')
            with patch.object(pipeline,'save_preview',wraps=pipeline.save_preview) as preview:
                main(path)
            self.assertEqual(len(preview.call_args_list),2)
            self.assertTrue(all(call.args[1].name=='water_polygon.tif' for call in preview.call_args_list))
            output=config['paths']['OUTPUT_DIR'];products=list(output.glob('*/*/water_polygon.*'))
            self.assertEqual(len(list(output.glob('*/*/terrain_quicklook.png'))),2)
            self.assertFalse(list(output.glob('*/*/water.tif')))
            self.assertFalse(list(output.glob('*/*/water.shp')))
            self.assertEqual(len(list(config['paths']['CACHE_DIR'].glob('scenes/*/products/terrain_exclusion.tif'))),2)
            before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in products if p.suffix!='.json'}
            with patch.object(pipeline,'segment',side_effect=AssertionError('must reuse')),patch.object(pipeline,'crop_org',side_effect=AssertionError('must reuse')):
                main(path)
            self.assertEqual(before,{p:hashlib.sha256(p.read_bytes()).hexdigest() for p in before})
            self.assertFalse(list(output.glob('*/*/water.tif')))
    def test_repair_uses_official_entry(self):
        path=fixtures.ROOT/'scripts/repair_gui_terrain.py'
        spec=importlib.util.spec_from_file_location('repair_entry',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertIs(module.main,main)

if __name__=='__main__':unittest.main()
