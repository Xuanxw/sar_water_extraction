"""把同一套四张专题图更新到已有的四图 DOCX 报告，保留正文和统计表。"""

import argparse
import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.oxml.ns import qn
from docx.shared import Cm
from PIL import Image


MAPS = ('01_water_change', '02_observation', '03_auxiliary', '04_facility_exposure')


def update(template, maps, output, settings):
    doc = Document(template)
    captions = settings.get('captions')
    max_heights = settings.get('max_image_height_cm', [12, 12, 10, 16])
    max_widths = settings.get('max_image_width_cm', [17.1, 17.1, 17.1, 22.7])
    if len(doc.inline_shapes) != len(MAPS):
        raise ValueError('请提供含四张内嵌图片的报告，顺序为变化、覆盖、辅助判读、设施暴露。')
    if page_size := settings.get('facility_page_mm'):
        if len(doc.sections) < 2:
            raise ValueError('调整设施图纸张时，末页需有独立的 Word 分节。')
        width, height = page_size
        last = doc.sections[-1]
        last.orientation = WD_ORIENT.PORTRAIT if height >= width else WD_ORIENT.LANDSCAPE
        last.page_width, last.page_height = Cm(width/10), Cm(height/10)
    for i, (picture, name) in enumerate(zip(doc.inline_shapes, MAPS)):
        path = maps / (name + '.png')
        blip = picture._inline.graphic.graphicData.pic.blipFill.blip
        old_rel = blip.get(qn('r:embed'))
        rel, _ = doc.part.get_or_add_image(str(path))
        blip.set(qn('r:embed'), rel)
        if old_rel != rel:
            doc.part.drop_rel(old_rel)
        with Image.open(path) as img:
            aspect = img.width / img.height
        height = min(max_heights[i], max_widths[i]/aspect)
        picture.height = Cm(height)
        picture.width = Cm(height*aspect)
        if captions:
            paragraph = next(p for p in doc.paragraphs if p.style.name == 'Caption' and p.text.startswith(f'图 {i+1} '))
            paragraph.text = captions[i]
            picture._inline.docPr.set('descr', captions[i])
    for old, new in settings.get('body_replacements', {}).items():
        paragraph = next(p for p in doc.paragraphs if old in p.text)
        run = next((r for r in paragraph.runs if old in r.text), None)
        if run is None:
            raise ValueError(f'正文替换需保持行内格式，请核对：{old}')
        run.text = run.text.replace(old, new)
    output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', type=Path)
    parser.add_argument('maps', type=Path)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    settings = json.loads(args.case.read_text(encoding='utf-8')).get('report', {})
    update(args.input, args.maps, args.output, settings)
