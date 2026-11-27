"""用本地 HTTP 分段响应验证续传、目录选择和发布校验。"""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Thread
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.data import download_s1s2_water as download


class S1S2DownloadTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("curl"), "需要系统 curl")
    def test_resume_after_repeated_partial_responses(self):
        data = bytes(range(32))
        offsets = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                offset = int(self.headers["Range"].removeprefix("bytes=").removesuffix("-"))
                offsets.append(offset)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {offset}-{len(data)-1}/{len(data)}")
                self.send_header("Content-Length", str(len(data) - offset))
                self.end_headers()
                # 每次只发送四字节就断线，完成本文件需要连续续传七次以上。
                self.wfile.write(data[offset:offset + 4])

            def log_message(self, *args):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server, TemporaryDirectory() as temp:
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                folder = Path(temp)
                partial = folder / "part1.zip.part"
                partial.write_bytes(data[:1])
                item = download.Download("part1.zip", len(data), "md5", hashlib.md5(data).hexdigest(),
                                         f"http://127.0.0.1:{server.server_port}/part1.zip")
                with patch.object(download.time, "sleep"), patch.dict("os.environ", no_proxy="127.0.0.1"):
                    download.download_file(folder, item, shutil.which("curl"))
                    download.download_file(folder, item, shutil.which("curl"))
                self.assertEqual(offsets, list(range(1, 32, 4)))
                self.assertEqual((folder / item.name).read_bytes(), data)
                self.assertFalse(partial.exists())
            finally:
                server.shutdown()
                thread.join()

    def test_stalled_download_stops_and_keeps_partial(self):
        item = download.Download("part1.zip", 10, "md5", "unused", "https://example.invalid")
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            partial = folder / "part1.zip.part"
            partial.write_bytes(b"abc")
            failed = subprocess.CompletedProcess([], 56, stdout="206")
            with patch.object(download.subprocess, "run", return_value=failed) as run, patch.object(download.time, "sleep"):
                with self.assertRaises(RuntimeError):
                    download.download_file(folder, item, "curl")
            self.assertEqual(run.call_count, download.MAX_STALLED)
            self.assertEqual(partial.read_bytes(), b"abc")
            self.assertFalse((folder / item.name).exists())

    def test_scene_layers_and_corrupt_partial(self):
        data = b"sample"
        sha = hashlib.sha256(data).hexdigest()
        blob = hashlib.sha1(b"blob 6\0sample").hexdigest()
        index = {f"59/sentinel12_s1_59_{layer}.tif": dict(size=6, lfs=dict(sha256=sha))
                 for layer in ("img", "msk", "valid")}
        index["59/sentinel12_59_meta.json"] = dict(size=6, blobId=blob)
        index["59/sentinel12_copdem30_59_elevation.tif"] = dict(size=6, lfs=dict(sha256=sha))
        files = download.s1_files(index, ["59", "59"])
        self.assertEqual(len(files), 4)
        self.assertEqual(len(download.s1_files(index, [], dem=True)), 5)
        elevations = download.s1_files(index, [], dem_only=True)
        self.assertEqual([item.name for item in elevations], ["59/sentinel12_copdem30_59_elevation.tif"])
        with self.assertRaises(ValueError):
            download.s1_files(index, ["2"])
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "59").mkdir()
            for item in files:
                partial = root / (item.name + ".part")
                partial.write_bytes(data)
                download.download_file(root, item, "unused")
                self.assertEqual((root / item.name).read_bytes(), data)
            item = files[0]
            (root / item.name).unlink()
            (root / (item.name + ".part")).write_bytes(b"broken")
            with self.assertRaisesRegex(RuntimeError, "sha256"):
                download.download_file(root, item, "unused")
            self.assertFalse((root / item.name).exists())
            self.assertEqual(next((root / "59").glob("*.bad-*")).read_bytes(), b"broken")
        del index[files[1].name]
        with self.assertRaisesRegex(ValueError, "缺少"):
            download.s1_files(index, [])


if __name__ == "__main__":
    unittest.main()
