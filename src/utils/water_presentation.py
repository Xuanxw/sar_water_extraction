"""水体成果三联图：只作显示，不改变预测、阈值或精度统计。"""
import gc
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Rectangle
from scipy.ndimage import binary_erosion


def save_water_presentation(image, prediction, path, title, input_caption="SAR input band 1"):
    """输出灰度底图、蓝色水体叠加、黄色框局部图。

    image/prediction为同网格H×W数组；水体=1，非水体=0，其余预测值未知。
    输入可以是训练概览，不能据此恢复原始细节。最长边限制1200像素，
    放大框只根据预测边缘选择，不使用真值挑选好案例。无地理信息时不画指北针或米制比例尺。
    """
    step = max(1, int(np.ceil(max(image.shape) / 1200)))
    sar = np.asarray(image[::step, ::step], dtype=np.float32)
    pred = np.asarray(prediction[::step, ::step])
    valid = np.isfinite(sar)
    known = valid & np.isin(pred, (0, 1))
    water = known & (pred == 1)
    values = sar[valid]
    low, high = np.percentile(values, (2, 98)) if values.size else (0., 1.)
    gray = np.zeros(sar.shape, dtype=np.float32)
    gray[valid] = np.clip((sar[valid] - low) / max(float(high-low), 1e-6), 0, 1)
    background = np.repeat((gray * 255).astype(np.uint8)[..., None], 3, axis=2)
    overlay = background.copy()
    overlay[water] = (35, 105, 230)

    height, width = sar.shape
    # 优先展示靠近图像中心的预测岸线；全水/无水时退回有效区域中心附近。
    candidates = water & ~binary_erosion(water, border_value=1)
    if not candidates.any():
        candidates = valid
    yy, xx = np.nonzero(candidates)
    if yy.size:
        index = np.argmin((yy-(height-1)/2)**2 + (xx-(width-1)/2)**2)
        cy, cx = int(yy[index]), int(xx[index])
    else:
        cy, cx = height//2, width//2
    crop_h, crop_w = max(1, height//3), max(1, width//3)
    top = min(max(0, cy-crop_h//2), height-crop_h)
    left = min(max(0, cx-crop_w//2), width-crop_w)
    crop = overlay[top:top+crop_h, left:left+crop_w]

    figure = Figure(figsize=(15, 5.8), facecolor="white")
    FigureCanvasAgg(figure)
    axes = axis = None
    try:
        axes = figure.subplots(1, 3)
        for axis, data, caption in zip(axes, (background, overlay, crop),
                (f"(a) {input_caption}", "(b) Predicted water (blue)", "(c) Yellow-box detail")):
            axis.imshow(data, interpolation="nearest")
            axis.set_title(caption, fontsize=11)
            axis.set_axis_off()
        axes[1].add_patch(Rectangle((left-.5, top-.5), crop_w, crop_h,
                                   fill=False, edgecolor="#ffd400", linewidth=1.6))
        figure.suptitle(title, fontsize=11, y=.97)
        note = " | No predicted water" if not water.any() else ""
        figure.text(.5, .055,
                    "P2-P98 display only | Overview crop; not native-resolution detail | "
                    "Black: no SAR data; unknown prediction left uncolored" + note,
                    ha="center", fontsize=9)
        figure.subplots_adjust(left=.015, right=.985, bottom=.11, top=.90, wspace=.025)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=120)
    finally:
        figure.clear()
        # 与现有训练图一致，及时释放Canvas循环引用，避免逐景绘图累积内存。
        del axes, axis
        del figure
        gc.collect()


def save_raster_water_presentation(image_path, mask_path, destination, max_side=1200,
                                  input_caption="Filtered SAR band 1"):
    """同网格SAR与最终0/1/NoData掩膜生成概览；只读缩略数组，不载入整景。

    保留栅格无效掩膜，不把NoData当水体；显示拉伸不改变文件像元或水体矢量。
    """
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(image_path) as image, rasterio.open(mask_path) as mask:
        if (image.shape, image.transform, image.crs) != (mask.shape, mask.transform, mask.crs):
            raise ValueError("展示图的SAR底图和水体掩膜必须位于同一网格")
        ratio = min(1., min(max_side, 1200) / max(image.shape))
        shape = (max(1, round(image.height*ratio)), max(1, round(image.width*ratio)))
        sar = image.read(1, out_shape=shape, out_dtype="float32", masked=True,
                         resampling=Resampling.nearest).filled(np.nan)
        water = mask.read(1, out_shape=shape, out_dtype="int16", masked=True,
                          resampling=Resampling.nearest).filled(-1)
    save_water_presentation(sar, water, destination,
                            f"{Path(mask_path).parent.name} | Final threshold water", input_caption)
