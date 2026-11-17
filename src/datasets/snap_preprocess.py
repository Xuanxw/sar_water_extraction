"""SNAP GPT 整景斑点滤波：BC训练ORG使用线性功率，不执行辐射校正。

由image_process.py显式生成独立缓存，训练/验证/推理只读复用。源影像和标签不改写。
仅临时补充真实的波段单位，不伪造SLC、卫星类型或校正LUT。
"""
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import hashlib
import json
import subprocess
import shutil
from time import perf_counter, sleep
import xml.etree.ElementTree as ET

from affine import Affine
from rasterio.crs import CRS
from filelock import FileLock, Timeout
import numpy as np
import rasterio

# 机器路径由image_process入口读取YAML后传入；底层没有隐式路径默认值。
GPT_PATH = None
CACHE_DIR = None
WORK_DIR = None
GPT_THREADS = 4
GPT_TILE_CACHE = '512M'
GPT_SCENE_WORKERS = 1
WRITER_SUFFIX = {"GeoTIFF-BigTIFF": ".tif", "BEAM-DIMAP": ".dim"}


def configure_paths(gpt_path, cache_dir, work_dir, *, threads=4, tile_cache='512M', scene_workers=1):
    """只接收执行层的SNAP运行配置，不启动处理、不创建目录。"""
    global GPT_PATH, CACHE_DIR, WORK_DIR, GPT_THREADS, GPT_TILE_CACHE, GPT_SCENE_WORKERS
    if type(scene_workers) is not int or scene_workers < 1:
        raise ValueError("scene_workers 必须为正整数")
    GPT_SCENE_WORKERS = scene_workers
    GPT_PATH, CACHE_DIR, WORK_DIR = Path(gpt_path), Path(cache_dir), Path(work_dir)
    GPT_THREADS, GPT_TILE_CACHE = int(threads), str(tile_cache)


@contextmanager
def gpt_slot():
    """每个 GPT 独享用户目录。第一个位置沿用串行锁，可与其他入口共用缓存。"""
    locks = [FileLock(str(CACHE_DIR / ('gpt.lock' if i == 0 else f'gpt_{i}.lock')))
             for i in range(GPT_SCENE_WORKERS)]
    while True:
        for i, lock in enumerate(locks):
            try:
                lock.acquire(timeout=0)
            except Timeout:
                continue
            try:
                yield CACHE_DIR / ('snap_userdir' if i == 0 else f'snap_userdir_{i}')
            finally:
                lock.release()
            return
        sleep(0.2)


@lru_cache(maxsize=4)
def snap_runtime(gpt_path):
    """记录实际计算模块的版本和指纹，升级SNAP后不复用旧统计量/滤波缓存。"""
    executable = Path(gpt_path)
    if not executable.is_file():
        raise FileNotFoundError(f'未找到SNAP GPT：{executable}；请检查本次YAML中的SNAP_GPT_PATH')
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


def dimap_band_vrt(product, expected_names=("power_1",)):
    """读取 BEAM .dim 声明的波段网格与 NoData。VRT 引用原始 .img，不重采样。"""
    doc = ET.parse(product)
    dims = doc.find("Raster_Dimensions")
    w, h = int(dims.findtext("NCOLS")), int(dims.findtext("NROWS"))
    if int(dims.findtext("NBANDS")) != len(expected_names):
        raise ValueError("BEAM 波段数与本次滤波输入不一致")
    # Java AffineTransform 的顺序是 m00,m10,m01,m11,m02,m12。
    a, d, b, e, c, f = map(float, doc.findtext("Geoposition/IMAGE_TO_MODEL_TRANSFORM").split(","))
    transform = Affine(a, b, c, d, e, f)
    crs = CRS.from_wkt(doc.findtext("Coordinate_Reference_System/WKT"))
    vrt = ET.Element("VRTDataset", rasterXSize=str(w), rasterYSize=str(h))
    ET.SubElement(vrt, "SRS").text = crs.to_wkt()
    ET.SubElement(vrt, "GeoTransform").text = ",".join(map(str, transform.to_gdal()))
    files = {int(node.findtext("BAND_INDEX")): node for node in doc.findall("Data_Access/Data_File")}
    bands = {int(node.findtext("BAND_INDEX")): node for node in doc.findall("Image_Interpretation/Spectral_Band_Info")}
    for index, name in enumerate(expected_names):
        band = bands[index]
        if band.findtext("BAND_NAME") != name or band.findtext("DATA_TYPE") != "float32":
            raise ValueError(f"BEAM 波段 {index + 1} 与预期 {name} (Float32) 不一致")
        header = files[index].find("DATA_FILE_PATH").attrib["href"]
        pixels = (product.parent / header).with_suffix(".img").resolve(strict=True)
        with rasterio.open(pixels) as ds:
            if (ds.width, ds.height, ds.count, ds.dtypes) != (w, h, 1, ("float32",)):
                raise ValueError("BEAM 波段文件与 .dim 声明不一致")
        layer = ET.SubElement(vrt, "VRTRasterBand", dataType="Float32", band=str(index + 1))
        if band.findtext("NO_DATA_VALUE_USED", "").lower() == "true":
            ET.SubElement(layer, "NoDataValue").text = str(float(band.findtext("NO_DATA_VALUE")))
        source = ET.SubElement(layer, "SimpleSource")
        ET.SubElement(source, "SourceFilename", relativeToVRT="1").text = pixels.relative_to(product.parent).as_posix()
        ET.SubElement(source, "SourceBand").text = "1"
    view = product.with_suffix(".vrt")
    ET.ElementTree(vrt).write(view, encoding="utf-8", xml_declaration=True)
    return view


class SnapSpeckleFilter:
    """普通Lee整景缓存；功率域滤波后恢复输入域，标准化由image_process.py后续模式负责。"""

    def __init__(self, config, *, writer_format="GeoTIFF-BigTIFF"):
        # 写出格式只决定临时产品，不能进入已发布 GeoTIFF 的数值签名。
        if writer_format not in WRITER_SUFFIX:
            raise ValueError(f"不支持的 SNAP 写出格式：{writer_format}")
        self.writer_format = writer_format
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
            raise FileNotFoundError(f'缺少匹配的SNAP预处理成果：{source.name}；请先运行 python src/datasets/image_process.py <配置.yaml>')
        folder.mkdir(parents=True, exist_ok=True)
        with FileLock(str(folder/'prepare.lock')):
            if not completed():
                print(f'SNAP Lee整景准备：{source.name}；输入域={self.config["input_domain"]}', flush=True)
                # 发布新影像之前移除旧完成标记；失败时保留日志，下一次重新处理本景。
                record.unlink(missing_ok=True)
                # 中间影像放SSD；完成后跨盘复制到I盘临时文件，再在同盘原子发布。
                # 工作目录不进入数值配置/缓存键，历史完成景仍可直接复用。
                WORK_DIR.mkdir(parents=True, exist_ok=True)
                with TemporaryDirectory(prefix='work_', dir=WORK_DIR) as temporary:
                    work = Path(temporary)
                    timings = self.run(source, bands, work/'filtered.tif', work, folder) or {}
                    started = perf_counter()
                    pending_image = folder/'filtered.tif.partial'
                    shutil.copyfile(work/'filtered.tif', pending_image)
                    pending_image.replace(output)
                    timings['cache_publish_seconds'] = round(perf_counter()-started, 3)
                    (folder/'timings.json').write_text(json.dumps(timings, indent=2), encoding='utf-8')
                    print(f'SNAP分阶段耗时（秒）：{timings}', flush=True)
                pending = record.with_suffix('.json.partial')
                pending.write_text(json.dumps(signature, ensure_ascii=False, indent=2), encoding='utf-8')
                pending.replace(record)
        self._prepared[key] = output
        return output

    def run(self, source, bands, output, work, diagnostics):
        """分窗整理有效域→GPT计算→恢复原网格/有效域；只在磁盘临时文件间传数据。"""
        timings = {}
        started = perf_counter()
        adapted = work/'input.tif'
        profile = self.prepare_input(source, bands, adapted)
        timings['input_preparation_seconds'] = round(perf_counter()-started, 3)
        graph_path = diagnostics/'graph.xml'
        raw_output = work / ('snap_output' + WRITER_SUFFIX[self.writer_format])
        self.write_graph(graph_path, adapted, raw_output, len(bands), format_name=self.writer_format)
        # 将Java输出写入日志，报错保留原始原因；不退回旧Lee继续训练。
        log = diagnostics/'gpt.log'
        started = perf_counter()
        with gpt_slot() as userdir, log.open('w', encoding='utf-8') as stream:
            userdir.mkdir(parents=True, exist_ok=True)
            command = [str(GPT_PATH), str(graph_path), '-c', GPT_TILE_CACHE, '-q', str(GPT_THREADS),
                       '-Dsnap.userdir=' + str(userdir)]
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        timings['gpt_seconds'] = round(perf_counter()-started, 3)
        if completed.returncode:
            raise RuntimeError(f'SNAP GPT失败，退出码{completed.returncode}；查看 {log}')

        # SNAP可能改变NoData标记；用原有效域恢复，合法0值仍有效，标签网格不变。
        started = perf_counter()
        if self.writer_format == "BEAM-DIMAP":
            prefix = "sar" if self.config["input_domain"] == "amplitude" else "power"
            names = [f"{prefix}_{i}" for i in range(1, len(bands) + 1)]
            raw_output = dimap_band_vrt(raw_output, names)
        self.restore_output(source, adapted, raw_output, output, profile, format_name=self.writer_format)
        timings['restore_mask_and_compress_seconds'] = round(perf_counter()-started, 3)
        return timings

    def prepare_input(self, source, bands, adapted):
        """按选定波段写入 GPT 共用的 Float32 输入，合法零保留，无效值编码为 -9999。"""
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
        return profile

    def restore_output(self, source, adapted, raw_output, output, profile, *, format_name="GeoTIFF-BigTIFF"):
        """校验原始网格，恢复输入有效域并无损压缩到同网格 GeoTIFF。"""
        profile = profile.copy()
        with rasterio.open(adapted) as original, rasterio.open(raw_output) as filtered:
            if filtered.shape != original.shape or filtered.count != profile['count']:
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
                        raise ValueError(f'SNAP在有效区域输出非法值；查看 {raw_output}')
                    dst.write(np.where(good[None], values, -9999).astype('float32'), window=window)
                dst.update_tags(input_domain=self.config['input_domain'], calibration='not_applied',
                                speckle='SNAP GPT Lee', snap_writer_format=format_name, source=str(source))
                for b in range(1, profile['count']+1):
                    dst.set_band_unit(b, self.config['input_domain'])

    def write_graph(self, path, source, output, count, *, format_name="GeoTIFF-BigTIFF"):
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
        # 线性功率可直接写出Lee结果。其他显式幅度工作流仍需恢复原单位。
        last_node = 'Lee'
        if amplitude:
            restored = ET.SubElement(node('Restore', 'BandMaths', 'Lee'), 'targetBands')
            for index in range(1, count+1):
                name = f'power_{index}'
                band = ET.SubElement(restored, 'targetBand')
                for key, value in dict(name=f'sar_{index}', type='float32', unit='amplitude',
                                       expression=f'{name} < 0 ? -9999 : sqrt({name})', noDataValue='-9999').items():
                    ET.SubElement(band, key).text = value
            last_node = 'Restore'
        write = node('Write', 'Write', last_node)
        ET.SubElement(write, 'file').text = str(output)
        ET.SubElement(write, 'formatName').text = format_name
        ET.indent(graph)
        ET.ElementTree(graph).write(path, encoding='utf-8', xml_declaration=True)

    def __call__(self, image, valid):
        """兼容旧切片集成推理的数组入口；整景训练使用prepare_file缓存，不走此慢路径。"""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with FileLock(str(CACHE_DIR/'array_diagnostics.lock')), TemporaryDirectory(prefix='array_', dir=CACHE_DIR) as temporary:
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
