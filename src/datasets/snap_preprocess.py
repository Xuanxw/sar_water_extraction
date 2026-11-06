"""SNAP GPT 整景斑点滤波：ORG已确认是线性幅度，不执行辐射校正。

由image_process.py显式生成独立缓存，训练/验证/推理只读复用。源影像和标签不改写。
仅临时补充真实的波段单位，不伪造SLC、卫星类型或校正LUT。
"""
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import hashlib
import json
import os
import subprocess
import xml.etree.ElementTree as ET

from filelock import FileLock
import numpy as np
import rasterio

# 环境变量允许换机器或试验目录；路径不属于模型的数值预处理参数。
GPT_PATH = Path(os.environ.get('SNAP_GPT', r'D:\SNAP\esa-snap\bin\gpt.exe'))
CACHE_DIR = Path(os.environ.get('SNAP_SPECKLE_CACHE', r'I:\水体数据\snap_lee_cache'))
GPT_THREADS = 4
GPT_TILE_CACHE = '512M'


def configure_paths(gpt_path, cache_dir):
    """两个入口显式设置工具和缓存位置，不执行任何影像处理。"""
    global GPT_PATH, CACHE_DIR
    GPT_PATH, CACHE_DIR = Path(gpt_path), Path(cache_dir)


@lru_cache(maxsize=4)
def snap_runtime(gpt_path):
    """记录实际计算模块的版本和指纹，升级SNAP后不复用旧统计量/滤波缓存。"""
    executable = Path(gpt_path)
    if not executable.is_file():
        raise FileNotFoundError(f'未找到SNAP GPT：{executable}；请设置SNAP_GPT')
    home = executable.parent.parent
    modules = [home/'snap/modules/org-esa-snap-snap-core.jar',
               home/'microwavetbx/modules/eu-esa-microwavetbx-sar-op-sar-processing.jar']
    result = {}
    for module in modules:
        with ZipFile(module) as archive:
            manifest = archive.read('META-INF/MANIFEST.MF').decode()
        version = next((line.split(': ', 1)[1] for line in manifest.splitlines()
                        if line.startswith('OpenIDE-Module-Specification-Version: ')), 'unknown')
        result[module.stem] = version + ':' + hashlib.sha256(module.read_bytes()).hexdigest()[:16]
    return result


class SnapSpeckleFilter:
    """普通Lee整景缓存；功率域滤波后恢复输入域，标准化由image_process.py后续模式负责。"""

    def __init__(self, config):
        self.config = dict(config)
        self._prepared = {}

    def prepare_file(self, source, bands, create=True):
        """create=True由独立预处理调用；create=False只校验/读取，不启动GPT。"""
        source = Path(source).resolve()
        stamps = [(str(source), source.stat().st_size, source.stat().st_mtime_ns)]
        if source.suffix.lower() == '.vrt':
            for node in ET.parse(source).findall('.//SourceFilename'):
                linked = Path(node.text)
                if node.get('relativeToVRT') == '1':
                    linked = source.parent/linked
                stamps.append((str(linked.resolve()), linked.stat().st_size, linked.stat().st_mtime_ns))
        signature = dict(sources=stamps, bands=list(bands), config=self.config)
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:24]
        if key in self._prepared:
            return self._prepared[key]
        folder = CACHE_DIR/f'{source.stem}_{key}'
        output, record = folder/'filtered.tif', folder/'processing.json'

        def completed():
            # 不仅检查文件名；中断、参数不匹配或损坏的完成记录不能当作成功。
            if not (output.is_file() and record.is_file()):
                return False
            try:
                receipt = json.loads(record.read_text(encoding='utf-8'))
                if json.dumps(receipt, sort_keys=True) != json.dumps(signature, sort_keys=True):
                    return False
                with rasterio.open(source) as src, rasterio.open(output) as dst:
                    return (src.shape == dst.shape and dst.count == len(bands)
                            and src.crs == dst.crs and src.transform.almost_equals(dst.transform))
            except (OSError, ValueError, rasterio.errors.RasterioError):
                return False

        if completed():
            self._prepared[key] = output
            return output
        if not create:
            raise FileNotFoundError(f'缺少匹配的SNAP预处理成果：{source.name}；请先运行 python src/datasets/image_process.py')
        folder.mkdir(parents=True, exist_ok=True)
        with FileLock(str(folder/'prepare.lock')):
            if not completed():
                print(f'SNAP Lee整景准备：{source.name}；输入域={self.config["input_domain"]}', flush=True)
                # 发布新影像之前移除旧完成标记；失败时保留日志，下一次重新处理本景。
                record.unlink(missing_ok=True)
                with FileLock(str(CACHE_DIR/'gpt.lock')):
                    with TemporaryDirectory(prefix='work_', dir=folder) as temporary:
                        work = Path(temporary)
                        self.run(source, bands, work/'filtered.tif', work, folder)
                        (work/'filtered.tif').replace(output)
                pending = record.with_suffix('.json.partial')
                pending.write_text(json.dumps(signature, ensure_ascii=False, indent=2), encoding='utf-8')
                pending.replace(record)
        self._prepared[key] = output
        return output

    def run(self, source, bands, output, work, diagnostics):
        """分窗整理有效域→GPT计算→恢复原网格/有效域；只在磁盘临时文件间传数据。"""
        adapted = work/'input.tif'
        with rasterio.open(source) as src:
            profile = dict(driver='GTiff', width=src.width, height=src.height, count=len(bands),
                           dtype='float32', crs=src.crs, transform=src.transform, nodata=-9999.,
                           tiled=True, blockxsize=256, blockysize=256, BIGTIFF='IF_SAFER')
            with rasterio.open(adapted, 'w', **profile) as dst:
                for _, window in dst.block_windows(1):
                    values = src.read(bands, window=window, masked=True, out_dtype='float32')
                    good = ~np.ma.getmaskarray(values).any(axis=0) & np.isfinite(values.data).all(axis=0)
                    if (values.data[:, good] < 0).any():
                        raise ValueError(f'线性SAR输入出现有效负值：{source}；请检查物理域/NoData')
                    limit = np.sqrt(np.finfo('float32').max) if self.config['input_domain'] == 'amplitude' else np.finfo('float32').max
                    if (values.data[:, good] > limit).any():
                        raise ValueError(f'输入值超出Float32功率范围：{source}')
                    dst.write(np.where(good[None], values.data, -9999).astype('float32'), window=window)

        graph_path, raw_output = diagnostics/'graph.xml', work/'snap_output.tif'
        self.write_graph(graph_path, adapted, raw_output, len(bands))
        userdir = CACHE_DIR/'snap_userdir'
        userdir.mkdir(parents=True, exist_ok=True)
        command = [str(GPT_PATH), str(graph_path), '-c', GPT_TILE_CACHE, '-q', str(GPT_THREADS),
                   '-Dsnap.userdir=' + str(userdir)]
        # 将Java输出写入日志，报错保留原始原因；不退回旧Lee继续训练。
        log = diagnostics/'gpt.log'
        with log.open('w', encoding='utf-8') as stream:
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if completed.returncode:
            raise RuntimeError(f'SNAP GPT失败，退出码{completed.returncode}；查看 {log}')

        # SNAP可能改变NoData标记；用原有效域恢复，合法0值仍有效，标签网格不变。
        with rasterio.open(adapted) as original, rasterio.open(raw_output) as filtered:
            if filtered.shape != original.shape or filtered.count != len(bands):
                raise ValueError('SNAP输出尺寸/波段数与输入不一致')
            # 旧切片数组入口没有地理参考；真实ORG有CRS时必须同时保持投影和仿射网格。
            if original.crs is not None and (filtered.crs != original.crs or not filtered.transform.almost_equals(original.transform)):
                raise ValueError('SNAP输出地理网格发生变化，不能直接对齐原标签')
            profile.update(compress='deflate', predictor=3)
            with rasterio.open(output, 'w', **profile) as dst:
                for _, window in dst.block_windows(1):
                    good = original.read_masks(1, window=window) > 0
                    values = filtered.read(window=window)
                    if not np.isfinite(values[:, good]).all() or (values[:, good] < 0).any():
                        raise ValueError(f'SNAP在有效区域输出非法值；查看 {log}')
                    dst.write(np.where(good[None], values, -9999).astype('float32'), window=window)
                dst.update_tags(input_domain=self.config['input_domain'], calibration='not_applied',
                                speckle='SNAP GPT Lee', source=str(source))
                for b in range(1, len(bands)+1):
                    dst.set_band_unit(b, self.config['input_domain'])

    def write_graph(self, path, source, output, count):
        """图中显式平方/开方；输入为功率时保持功率，不做dB或辐射校正。"""
        graph = ET.Element('graph', id='ORG_SNAP_Lee')
        ET.SubElement(graph, 'version').text = '1.0'

        def node(name, operator, parent=None):
            element = ET.SubElement(graph, 'node', id=name)
            ET.SubElement(element, 'operator').text = operator
            if parent:
                ET.SubElement(ET.SubElement(element, 'sources'), 'sourceProduct', refid=parent)
            return ET.SubElement(element, 'parameters')

        read = node('Read', 'Read')
        ET.SubElement(read, 'file').text = str(source)
        power = ET.SubElement(node('Power', 'BandMaths', 'Read'), 'targetBands')
        amplitude = self.config['input_domain'] == 'amplitude'
        for index in range(1, count+1):
            # 自建无描述GeoTIFF由SNAP读取为band_1、band_2等。
            name = f'band_{index}'
            band = ET.SubElement(power, 'targetBand')
            expression = f'{name} * {name}' if amplitude else name
            for key, value in dict(name=f'power_{index}', type='float32', unit='intensity',
                                   expression=f'{name} == -9999 ? -9999 : {expression}',
                                   noDataValue='-9999').items():
                ET.SubElement(band, key).text = value
        lee = node('Lee', 'Speckle-Filter', 'Power')
        for key, value in dict(filter='Lee', filterSizeX=self.config['window_size'],
                               filterSizeY=self.config['window_size'], estimateENL='false',
                               enl=self.config['enl']).items():
            ET.SubElement(lee, key).text = str(value)
        restored = ET.SubElement(node('Restore', 'BandMaths', 'Lee'), 'targetBands')
        for index in range(1, count+1):
            name = f'power_{index}'
            expression = f'sqrt({name})' if amplitude else name
            band = ET.SubElement(restored, 'targetBand')
            for key, value in dict(name=f'sar_{index}', type='float32', unit=self.config['input_domain'],
                                   expression=f'{name} < 0 ? -9999 : {expression}', noDataValue='-9999').items():
                ET.SubElement(band, key).text = value
        write = node('Write', 'Write', 'Restore')
        ET.SubElement(write, 'file').text = str(output)
        ET.SubElement(write, 'formatName').text = 'GeoTIFF-BigTIFF'
        ET.indent(graph)
        ET.ElementTree(graph).write(path, encoding='utf-8', xml_declaration=True)

    def __call__(self, image, valid):
        """兼容旧切片集成推理的数组入口；整景训练使用prepare_file缓存，不走此慢路径。"""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with FileLock(str(CACHE_DIR/'gpt.lock')), TemporaryDirectory(prefix='array_', dir=CACHE_DIR) as temporary:
            work = Path(temporary)
            diagnostics = CACHE_DIR/'array_diagnostics'
            diagnostics.mkdir(exist_ok=True)
            source = work/'array.tif'
            with rasterio.open(source, 'w', driver='GTiff', height=image.shape[1], width=image.shape[2],
                               count=image.shape[0], dtype='float32', nodata=-9999.,
                               transform=rasterio.transform.from_origin(0, image.shape[1], 1, 1)) as dst:
                dst.write(np.where(valid[None], image, -9999).astype('float32'))
            self.run(source, tuple(range(1, image.shape[0]+1)), work/'result.tif', work, diagnostics)
            with rasterio.open(work/'result.tif') as src:
                return np.where(valid[None], src.read(), 0).astype('float32')
