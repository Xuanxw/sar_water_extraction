"""将 S1 Sigma0 dB 和可选高程配准到原影像网格，输出整景水体掩膜。"""
import math
from contextlib import ExitStack, closing
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import rasterio
from rasterio.enums import MaskFlags
from rasterio.windows import Window

from datasets.copernicus_dem import aligned_dem_tiles, read_elevation
from datasets.relative_normalization import RELATIVE_REPRESENTATIONS
from datasets.dfc_sample.s1_dataset import relative_backscatter_ranks, vv_rank_dem_m
from supervised.evaluation.sliding_inference import iter_scene_predictions
from utils.raster import windows


def predict_grd_scene(model, source, destination, *, mean, std, patch_representation,
                      bands, dem_path=None, threshold=0.5, device="cpu", window_size=512,
                      halo=64, batch_size=1):
    """按训练通道和窗口秩表示输出整景 uint8 水体掩膜（0 陆、1 水、255 无效）。

    DFC 波段 1=VV、2=VH、4=Copernicus DEM（米）。密云输入按实际 Sigma0 波段描述
    识别极化。VV、VH 同时精确为 0 dB 是该批数据确认的未覆盖编码。
    DEM 模式保留米值至训练统计量标准化。重叠窗按高斯权重融合 logits 再阈值化。
    """
    if patch_representation in RELATIVE_REPRESENTATIONS:
        raise ValueError("训练集 P02/P98 参数属于 DFC 存储值，不能直接应用于 Sigma0 dB 整景；"
                         "须先核实两者的数值映射，不能回退到逐窗秩归一化。")
    mixed = patch_representation == "vv_rank_dem_m"
    if mixed:
        if tuple(bands) != (1, 4) or dem_path is None:
            raise ValueError("VV+DEM 检查点应使用 DFC 波段 (1,4)，并提供米制 Copernicus DEM")
    elif patch_representation != "relative_backscatter_ranks" or tuple(bands) != (1, 2):
        raise ValueError("仅支持 VV+DEM 或 VV/VH 的分块秩表示检查点")
    if not 0 < threshold < 1:
        raise ValueError("水体概率阈值必须在 (0,1) 内")
    mean = np.asarray(mean, dtype=np.float32).reshape(-1)
    std = np.asarray(std, dtype=np.float32).reshape(-1)
    if mean.size != 2 or std.size != 2 or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("检查点需提供两个通道的有限均值和正标准差")
    source, destination = Path(source), Path(destination)
    if destination.resolve() == source.resolve() or (mixed and destination.resolve() == Path(dem_path).resolve()):
        raise ValueError("预测输出不能覆盖输入数据")
    logit_threshold = math.log(threshold / (1 - threshold))

    def transform_window(image, valid):
        transform = vv_rank_dem_m if mixed else relative_backscatter_ranks
        return transform(image, valid), valid

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as staging:
        inputs = Path(staging) / "model_inputs.tif"
        pending = Path(staging) / "mask.tif"
        with ExitStack() as stack:
            src = stack.enter_context(rasterio.open(source))
            if src.crs is None:
                raise ValueError(f"缺少 S1 地理参考：{source}")
            descriptions = [(d or "").lower() for d in src.descriptions]
            if descriptions.count("sigma0_vv_db") != 1 or descriptions.count("sigma0_vh_db") > 1:
                raise ValueError(f"无法唯一定位 Sigma0 dB 极化：{descriptions}")
            vv = descriptions.index("sigma0_vv_db") + 1
            dual = "sigma0_vh_db" in descriptions
            if dual:
                indices = (vv, descriptions.index("sigma0_vh_db") + 1)
            else:
                if not mixed or MaskFlags.all_valid in src.mask_flag_enums[vv - 1]:
                    raise ValueError("单 VV 产品仅可用于 VV+DEM 检查点，且必须声明有效覆盖掩膜或 NoData")
                indices = (vv,)
            profile = dict(driver="GTiff", width=src.width, height=src.height, crs=src.crs,
                           transform=src.transform, tiled=True, BIGTIFF="IF_SAFER")
            tiles = aligned_dem_tiles(dem_path, src, stack) if mixed else []
            usable = 0
            print(f"{source.name}：准备共网格输入与覆盖掩膜", flush=True)
            with rasterio.open(inputs, "w", **profile, count=2, dtype="float32", nodata=float("nan")) as aligned:
                for window in windows(src, 512):
                    sar = src.read(indices, window=window, masked=True, out_dtype="float32")
                    good = ~np.ma.getmaskarray(sar).any(axis=0) & np.isfinite(sar.data).all(axis=0)
                    if dual:
                        good &= ~np.all(sar.data == 0, axis=0)
                    image = sar.data
                    if mixed:
                        elevation = read_elevation(tiles, window)
                        image = np.stack((sar.data[0], elevation))
                        good &= np.isfinite(elevation)
                    usable += int(good.sum())
                    image[:, ~good] = np.nan
                    aligned.write(image, window=window)
            if not usable:
                raise ValueError(f"S1 与所需输入没有共同有效像元：{source}")
        stride = max(1, (window_size - 2 * halo) // 2)
        blocks = iter_scene_predictions(
            model, inputs, bands=(1, 2), window_size=window_size, halo=halo, stride=stride,
            blend="gaussian", mean=mean, std=std, device=device, batch_size=batch_size,
            boundary_context=False, window_transform=transform_window)
        with rasterio.open(pending, "w", **profile, count=1, dtype="uint8", nodata=255, compress="deflate") as dst, closing(blocks):
            dst.set_band_description(1, "water_mask")
            dst.update_tags(input_channels="VV,DEM_m" if mixed else "VV,VH", dem_source=str(dem_path or ""),
                            patch_representation=patch_representation, invalid="GDAL mask, nonfinite" + (", or VV=VH=0 dB" if dual else ""),
                            probability_threshold=threshold, window_size=window_size, halo=halo, stride=stride)
            next_progress = 10
            for block in blocks:
                top, left, height, width = block["window"]
                mask = np.where(block["valid_mask"], block["scores"] >= logit_threshold, 255).astype("uint8")
                dst.write(mask, 1, window=Window(left, top, width, height))
                progress = 100 * (top + height) // dst.height
                if progress >= next_progress and left + width == dst.width:
                    print(f"{source.name}：已写入 {top + height}/{dst.height} 行（{progress}%）", flush=True)
                    next_progress = 10 * (1 + progress // 10)
        pending.replace(destination)
    return destination
