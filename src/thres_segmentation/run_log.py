"""阈值生产日志：控制台着色、文本日志与阶段耗时同步追加，不参与算法缓存签名。"""
import csv
import re
import sys
import traceback
import warnings
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from datetime import datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4
from zoneinfo import ZoneInfo

from termcolor import colored


class LogStream:
    """将已有print和Python警告接入同一日志；保留原控制台，避免递归写入。"""
    def __init__(self, log, level='INFO'):
        self.log,self.level,self.pending=log,level,''

    def write(self,text):
        self.pending+=text
        while '\n' in self.pending:
            line,self.pending=self.pending.split('\n',1)
            if line.strip():self.log.message(line.rstrip('\r'),self.level)
        return len(text)

    def flush(self):
        if self.pending:
            self.log.message(self.pending,self.level);self.pending=''

    def isatty(self):
        return False


class RunLog:
    """每次启动分配run_id；每条记录立即flush，中断后再次运行追加而非覆盖。"""
    COLORS={'INFO':'white','START':'cyan','DONE':'green','REUSED':'yellow',
            'SKIPPED':'yellow','WARNING':'yellow','ERROR':'red','INTERRUPTED':'magenta'}
    STATUS={'completed':'DONE','reused':'REUSED','skipped':'SKIPPED',
            'failed':'ERROR','interrupted':'INTERRUPTED'}
    FIELDS=('run_id','scene_id','stage','name','event','status','started_at','ended_at','elapsed_seconds')

    def __init__(self,root,console_color=True):
        self.root=Path(root);self.console_color=console_color
        self.run_id=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:8]
        self.console=sys.stdout;self.frames=[]

    @staticmethod
    def now():
        return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='milliseconds')

    def message(self,text,level='INFO'):
        """控制台保留颜色，文件只写纯文本；下层下载和SNAP的print也走这里。"""
        text=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',str(text))
        stage,scene=self.frames[-1] if self.frames else ('RUN','')
        prefix=f'{self.now()} [{self.run_id}] [{stage}] [{level}]'
        if scene:prefix+=f' [{scene}]'
        for line in text.splitlines():
            value=f'{prefix} {line}'
            self.file.write(value+'\n');self.file.flush()
            display=colored(value,self.COLORS.get(level,'white'),force_color=True) if self.console_color else value
            self.console.write(display+'\n');self.console.flush()

    def record(self,scene,stage,name,event,status,started,ended='',elapsed=''):
        self.writer.writerow((self.run_id,scene,stage,name,event,status,started,ended,elapsed))
        self.timing_file.flush()

    @contextmanager
    def stage(self,number,name,scene='',status='completed'):
        """开始事件先落盘；结束事件含本次墙钟耗时，父阶段耗时包含子阶段，不能重复求和。"""
        state={'status':status};started=self.now();clock=perf_counter()
        self.frames.append((number,scene))
        self.record(scene,number,name,'start','running',started)
        self.message(f'{name}：开始','START')
        try:
            yield state
        except BaseException as exc:
            state['status']='interrupted' if isinstance(exc,(KeyboardInterrupt,SystemExit)) else 'failed'
            raise
        finally:
            elapsed=round(perf_counter()-clock,3);ended=self.now()
            self.record(scene,number,name,'end',state['status'],started,ended,elapsed)
            self.message(f'{name}：{state["status"]}；耗时 {elapsed:.3f} 秒',self.STATUS[state['status']])
            self.frames.pop()

    @contextmanager
    def capture(self):
        """只作用于本次同步任务；退出时恢复stdout/stderr，完整异常写入processing.log。"""
        self.root.mkdir(parents=True,exist_ok=True)
        timing=self.root/'stage_timings.csv';header=not timing.exists() or timing.stat().st_size==0
        with (self.root/'processing.log').open('a',encoding='utf-8') as self.file, \
             timing.open('a',encoding='utf-8-sig',newline='') as self.timing_file:
            self.writer=csv.writer(self.timing_file)
            if header:self.writer.writerow(self.FIELDS);self.timing_file.flush()
            out,err=LogStream(self),LogStream(self,'ERROR')
            def show_warning(message, category, filename, lineno, file=None, line=None):
                # Python warning单独标级；stderr中的异常仍为ERROR，不隐藏警告。
                self.message(f'{filename}:{lineno}: {category.__name__}: {message}', 'WARNING')
            with redirect_stdout(out),redirect_stderr(err),warnings.catch_warnings():
                warnings.showwarning=show_warning
                self.message(f'本次运行日志：{self.root / "processing.log"}')
                try:
                    yield self
                except BaseException as exc:
                    self.message(traceback.format_exc(),'INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'ERROR')
                    raise
                finally:
                    out.flush();err.flush()
