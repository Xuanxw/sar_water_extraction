"""公共工具的字节兼容性、原子写入和栅格分块契约。"""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from utils.files import sha256,write_json,stamp,signature
from utils.raster import windows


class SharedUtilsTests(unittest.TestCase):
    def test_file_hash_and_stamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'影像.bin';path.write_bytes(b'abc')
            self.assertEqual(sha256(path),'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')
            self.assertEqual(stamp(path),dict(path=str(path.resolve()),size=3,mtime_ns=path.stat().st_mtime_ns))

    def test_json_failure_keeps_previous_complete_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'nested/记录.json'
            write_json(path,{'状态':'完整'})
            before=path.read_bytes()
            with self.assertRaises(ValueError):write_json(path,{'value':float('nan')})
            self.assertEqual(path.read_bytes(),before)
            write_json(path,{'path':Path('dem.tif')},default=str)
            self.assertEqual(json.loads(path.read_text(encoding='utf8')),{'path':'dem.tif'})
            write_json(path,{'value':float('nan')},allow_nan=True)
            self.assertIn('NaN',path.read_text())
            self.assertFalse(path.with_suffix('.json.partial').exists())

    def test_cache_signature_is_byte_compatible(self):
        value={'影像':Path('dem.tif'),'bands':[1,2],'threshold':25.0}
        expected=hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()[:20]
        self.assertEqual(signature(value),expected)

    def test_edge_windows_cover_once(self):
        result=list(windows(SimpleNamespace(width=5,height=3),2))
        self.assertEqual([(w.col_off,w.row_off,w.width,w.height) for w in result],
                         [(0,0,2,2),(2,0,2,2),(4,0,1,2),(0,2,2,1),(2,2,2,1),(4,2,1,1)])


if __name__=='__main__':unittest.main()
