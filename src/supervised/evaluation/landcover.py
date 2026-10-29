"""读取与合并土地覆盖水体，供阈值流程及模型后处理共用。"""
import numpy as np
import rasterio


def merge_landcover(prediction, landcover, land_valid, image_valid, water_class=80):
    """补入有效的土地覆盖水体像素；非水体类别不强制删除模型预测的洪水。"""
    return (prediction | (land_valid & (landcover == water_class))) & image_valid


def read_landcover(image_path, band=5):
    """读取同文件中的土地覆盖层，返回数值与有效掩膜；供后处理和错误分析共用。"""
    with rasterio.open(image_path) as source:
        cover = source.read(band, masked=True)
    values = cover.compressed()
    codes = set(np.unique(values[np.isfinite(values)]).tolist())
    allowed = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100, 255}
    if not codes <= allowed:
        raise ValueError(f"{image_path} 第 {band} 波段不符合 WorldCover 编码：{codes - allowed}")
    valid = (~np.ma.getmaskarray(cover) & np.isfinite(cover.data)
             & ~np.isin(cover.data, (0, 255)))
    return cover.data.astype(np.int64), valid


