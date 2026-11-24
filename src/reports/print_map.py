"""把固定版式专题图等比例放入指定纸张，保留 PDF 中的矢量线和文字。"""

from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps


def fit_print_map(source, destination, *, width_mm, height_mm, dpi):
    """按纸张毫米尺寸等比缩放源 PDF 和 PNG，居中留出可能产生的空边。"""
    from pypdf import PdfReader, PdfWriter, Transformation
    from pypdf.generic import RectangleObject

    source, destination = Path(source), Path(destination)
    pdf = PdfReader(BytesIO(source.with_suffix('.pdf').read_bytes()))
    writer = PdfWriter(clone_from=pdf)
    page = writer.pages[0]
    width, height = width_mm * 72 / 25.4, height_mm * 72 / 25.4
    box = page.mediabox
    scale = min(width / float(box.width), height / float(box.height))
    x = (width - float(box.width) * scale) / 2 - float(box.left) * scale
    y = (height - float(box.height) * scale) / 2 - float(box.bottom) * scale
    page.add_transformation(Transformation().scale(scale).translate(x, y))
    page.mediabox = RectangleObject((0, 0, width, height))
    page.cropbox = RectangleObject((0, 0, width, height))

    size = round(width_mm * dpi / 25.4), round(height_mm * dpi / 25.4)
    with Image.open(source.with_suffix('.png')) as original:
        fitted = ImageOps.contain(original.convert('RGB'), size, Image.Resampling.LANCZOS)
    paper = Image.new('RGB', size, 'white')
    paper.paste(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
    writer.write(destination.with_suffix('.pdf'))
    paper.save(destination.with_suffix('.png'), dpi=(dpi, dpi))
