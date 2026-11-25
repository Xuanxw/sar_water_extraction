"""同网格显示 Sen1Floods11 人工标注组的 VV、真彩色、GT 及水体边界。"""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont


UNKNOWN = (150, 83, 160)
BOUNDARY = (0, 202, 255)


def _check_grid(reference, other):
    """同 CRS、同尺寸，四角位置的差异应小于十万分之一像元。"""
    if reference.crs is None or other.crs != reference.crs or other.shape != reference.shape:
        raise ValueError(f"影像、标签的尺寸或坐标系不一致：{other.name}")
    to_reference = ~reference.transform * other.transform
    for corner in ((0, 0), (other.width, 0), (0, other.height), (other.width, other.height)):
        if not np.allclose(to_reference * corner, corner, rtol=0, atol=1e-5):
            raise ValueError(f"影像、标签的像元网格不一致：{other.name}")


def _display_rgb(bands, low, high):
    """按固定端点转为 8 位 RGB。输入是掩膜数组 [1或3,H,W]，输出独立表示无效区。"""
    values = bands.filled(low).astype(np.float32)
    valid = np.all(~np.ma.getmaskarray(bands) & np.isfinite(values), axis=0)
    values[:, ~valid] = low
    scaled = np.rint(255 * np.clip((values - low) / (high - low), 0, 1)).astype(np.uint8)
    if len(scaled) == 1:
        scaled = np.repeat(scaled, 3, axis=0)
    rgb = np.moveaxis(scaled, 0, -1).copy()
    rgb[~valid] = UNKNOWN
    return rgb, valid


def _water_edges(water, known):
    """在两侧标签均已判读的四邻域提取水陆边界，忽略未知区和图块外缘。"""
    edges = np.zeros_like(water)
    vertical = known[1:] & known[:-1] & (water[1:] != water[:-1])
    horizontal = known[:, 1:] & known[:, :-1] & (water[:, 1:] != water[:, :-1])
    edges[1:] |= vertical
    edges[:-1] |= vertical
    edges[:, 1:] |= horizontal
    edges[:, :-1] |= horizontal
    return edges


def render_quicklook(s1: Path, s2: Path, gt: Path, output: Path):
    """输出 2×2 原尺寸 PNG。VV 按 dB 显示，RGB 使用 TOA 反射率，GT 采用 -1/0/1。"""
    s1, s2, gt, output = map(Path, (s1, s2, gt, output))
    sample = s1.stem.removesuffix("_S1Hand")
    if (s1.stem, s2.stem, gt.stem) != (f"{sample}_S1Hand", f"{sample}_S2Hand", f"{sample}_LabelHand"):
        raise ValueError(f"S1、S2、GT 的样本编号不匹配：{s1.name}, {s2.name}, {gt.name}")
    with rasterio.open(s1) as radar, rasterio.open(s2) as optical, rasterio.open(gt) as label:
        _check_grid(radar, optical)
        _check_grid(radar, label)
        if (radar.count, optical.count, label.count) != (2, 13, 1):
            raise ValueError(f"{sample} 的 S1、S2、GT 波段数不符合官方 2/13/1 定义")
        # v1.1 的 S1 已经是 dB。S2 L1C 存储反射率乘 10000；线性显示 0 至 0.30。
        vv_rgb, vv_valid = _display_rgb(radar.read([1], masked=True), -25.0, 0.0)
        rgb, _ = _display_rgb(optical.read([4, 3, 2], masked=True), 0, 3000)
        raw = label.read(1, masked=True)
    gt_valid = ~np.ma.getmaskarray(raw)
    if not np.isin(raw.data[gt_valid], (-1, 0, 1)).all():
        raise ValueError(f"{sample} 含超出 -1/0/1 的有效 GT 编码")
    known = gt_valid & (raw.data != -1)
    water = raw.data == 1
    gt_rgb = np.full(vv_rgb.shape, UNKNOWN, dtype=np.uint8)
    gt_rgb[known & ~water] = 255
    gt_rgb[known & water] = 0
    overlay = vv_rgb.copy()
    overlay[_water_edges(water, known) & vv_valid] = BOUNDARY

    height, width = known.shape
    pad, gap, heading = 14, 12, 32
    y0 = 45
    row_height = heading + height + gap
    canvas = Image.new("RGB", (2 * pad + 2 * width + gap, y0 + 2 * row_height + 6), "white")
    draw = ImageDraw.Draw(canvas)
    font_file = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "msyh.ttc"
    chinese = font_file.is_file()
    font_name = str(font_file) if chinese else "DejaVuSans.ttf"
    title_font = ImageFont.truetype(font_name, 21)
    panel_font = ImageFont.truetype(font_name, 16)
    draw.text((pad, 6), f"Sen1Floods11  /  {sample}", fill=(25, 36, 45), font=title_font)
    legend = (("GT 水体边界", "未知 / 无效") if chinese else ("GT boundary", "Unknown / invalid"))
    legend_x = canvas.width - (255 if chinese else 300)
    for color, caption in zip((BOUNDARY, UNKNOWN), legend):
        draw.rectangle((legend_x, 15, legend_x + 12, 27), fill=color)
        draw.text((legend_x + 19, 8), caption, fill=(32, 45, 54), font=panel_font)
        legend_x += 125 if chinese else 137
    headings = (("S1 VV（−25 至 0 dB）", "S2 L1C 真彩色（B4/B3/B2，反射率 0–0.30）",
                 "人工 GT（水体黑，非水白）", "VV 叠加人工 GT 水体边界") if chinese else
                ("S1 VV (-25 to 0 dB)", "S2 L1C RGB (B4/B3/B2, reflectance 0-0.30)",
                 "Hand GT (water black, non-water white)", "VV with hand-labelled water boundary"))
    for i, (caption, array) in enumerate(zip(headings, (vv_rgb, rgb, gt_rgb, overlay))):
        x = pad + (i % 2) * (width + gap)
        y = y0 + (i // 2) * row_height
        draw.text((x, y), caption, fill=(32, 45, 54), font=panel_font)
        canvas.paste(Image.fromarray(array), (x, y + heading))
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, prefix=output.stem + ".", suffix=".tmp", delete=False) as f:
        temporary = Path(f.name)
    try:
        canvas.save(temporary, format="PNG", compress_level=3)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
