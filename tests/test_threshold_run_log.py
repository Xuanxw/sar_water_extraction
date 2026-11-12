"""验证同步落盘、颜色隔离、异常与追加运行；仅使用临时目录。"""
import csv
import io
import sys
import unittest
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from thres_segmentation.run_log import RunLog


class RunLogTests(unittest.TestCase):
    def test_sync_append_color_and_failure(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);console=io.StringIO()
            with redirect_stdout(console):
                log=RunLog(root,console_color=True)
                with log.capture(),log.stage('04','测试裁剪','scene-A'):
                    print('来自下层print的进度',flush=True)
                    self.assertIn('来自下层print的进度',(root/'processing.log').read_text(encoding='utf-8'))
                    with (root/'stage_timings.csv').open(encoding='utf-8-sig',newline='') as src:
                        self.assertEqual(list(csv.DictReader(src))[-1]['event'],'start')
                previous=(root/'processing.log').read_bytes()
                log2=RunLog(root,console_color=False)
                with self.assertRaisesRegex(RuntimeError,'模拟失败'):
                    with log2.capture(),log2.stage('08','测试矢量','scene-B'):
                        raise RuntimeError('模拟失败')
            text=(root/'processing.log').read_text(encoding='utf-8')
            self.assertTrue((root/'processing.log').read_bytes().startswith(previous))
            self.assertNotIn('\x1b[',text)
            self.assertIn('\x1b[',console.getvalue())
            self.assertIn('Traceback',text);self.assertIn('模拟失败',text)
            with (root/'stage_timings.csv').open(encoding='utf-8-sig',newline='') as src:
                rows=list(csv.DictReader(src))
            self.assertEqual(len(rows),4)
            self.assertEqual(rows[-1]['status'],'failed')
            self.assertNotEqual(rows[0]['run_id'],rows[-1]['run_id'])
            self.assertGreaterEqual(float(rows[-1]['elapsed_seconds']),0)
            self.assertIn('+08:00',rows[-1]['started_at'])

    def test_keyboard_interrupt_records_status_and_restores_stream(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);original=sys.stdout
            with redirect_stdout(io.StringIO()):
                with self.assertRaises(KeyboardInterrupt):
                    with RunLog(root,console_color=False).capture() as log:
                        with log.stage('05','SNAP','scene-A'):raise KeyboardInterrupt()
            self.assertIs(sys.stdout,original)
            with (root/'stage_timings.csv').open(encoding='utf-8-sig',newline='') as src:
                self.assertEqual(list(csv.DictReader(src))[-1]['status'],'interrupted')

    def test_warning_level_color_and_hook_restored(self):
        original=warnings.showwarning
        with TemporaryDirectory() as temp,redirect_stdout(io.StringIO()) as console:
            root=Path(temp)
            with RunLog(root,console_color=True).capture():
                with warnings.catch_warnings():
                    warnings.simplefilter('always')
                    warnings.warn('旧接口提示',DeprecationWarning)
                print('真正的stderr错误',file=sys.stderr)
            text=(root/'processing.log').read_text(encoding='utf8')
            lines=[line for line in text.splitlines() if '旧接口提示' in line]
            self.assertEqual(len(lines),1)
            self.assertIn('[WARNING]',lines[0]);self.assertNotIn('[ERROR]',lines[0])
            self.assertTrue(any('[ERROR]' in line and '真正的stderr错误' in line for line in text.splitlines()))
            self.assertIn('\x1b[33m',console.getvalue())
            self.assertIs(warnings.showwarning,original)


if __name__=='__main__':unittest.main()
