"""Append portrait A4 atlas pages to the project emergency report."""
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


def style_run(run, *, size, bold=False, color="172C3A"):
    run.font.name = "Microsoft YaHei"
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor.from_string(color)
    return run


def compact(paragraph, *, line=None, before=0, after=0, keep=False):
    fmt = paragraph.paragraph_format
    fmt.space_before = Pt(before)
    fmt.space_after = Pt(after)
    fmt.keep_with_next = keep
    if line is not None:
        fmt.line_spacing = Pt(line)
    else:
        fmt.line_spacing = 1
    return paragraph


def legend(doc, facilities):
    items = [
        ("━", "道路相交段", "AA2835"),
        ("┄", "带桥隧标记的路段", "6B1823"),
        ("■", "新增水面", "E28616"),
        ("■", "稳定水面", "347FAA"),
        ("■", "减少水面", "7067A2"),
    ]
    sources = [spec["label"] for spec in facilities if spec["kind"] == "building"]
    buildings = []
    if sources:
        buildings.extend([("■", f"{sources[0]} 建筑重叠 ≥50%", "AE144F"),
                          ("■", f"{sources[0]} 建筑重叠 <50%", "D36883")])
    if len(sources) > 1:
        buildings.append(("■", f"{sources[1]} 建筑相交", "116964"))
    for group in (items[:2], buildings, items[2:]):
        p = compact(doc.add_paragraph(), line=13.2, after=1)
        for mark, label, color in group:
            style_run(p.add_run(mark+" "), size=9.2, bold=True, color=color)
            style_run(p.add_run(label+"    "), size=9.0)


def picture(doc, file, width, alt):
    p = compact(doc.add_paragraph(), after=1.5)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    shape = p.add_run().add_picture(str(file), width=Cm(width))
    shape._inline.docPr.set("descr", alt)


def heading(doc, name, *, top=False):
    p = compact(doc.add_heading(name, level=1), line=21, after=3, keep=True)
    p.paragraph_format.page_break_before = top
    style_run(p.runs[0], size=15.5, bold=True, color="000000")
    return p


def append_atlas(template, directory, output, config, manifest):
    """在正文末尾附加 A4 竖向详图。基报告按四图报告模板排版。"""
    maps, counts = manifest["maps"], manifest["counts"]
    doc = Document(template)
    if len(doc.inline_shapes) != 4:
        raise ValueError("详图附录的输入必须是已排版的四图应急报告。")
    for paragraph in doc.paragraphs:
        for run in paragraph.runs:
            if "道路和建筑候选位置见图 4。" in run.text:
                run.text = run.text.replace("道路和建筑候选位置见图 4。",
                                            "图 4 为总览，附录 A 收录全部候选位置的局部图。" if maps
                                            else "图 4 为设施核查总览。")

    last_section = doc.sections[-1]
    if last_section.page_width > last_section.page_height:
        from docx.enum.section import WD_ORIENT, WD_SECTION_START
        section = doc.add_section(WD_SECTION_START.NEW_PAGE)
        section.orientation = WD_ORIENT.PORTRAIT
        section.page_width, section.page_height = Cm(21), Cm(29.7)
        top = False
    else:
        top = True
    heading(doc, "附录 A  道路与建筑潜在暴露详图", top=top)
    p = compact(doc.add_paragraph(), line=14.5, after=4)
    if not maps:
        style_run(p.add_run("在本期两期共同有效观测区内，设施底图与新增水面未检出正长度或正面积重叠。"),
                  size=10.0)
        doc.save(output)
        return
    km = sum(detail["road_m"] for detail in maps)/1000
    road_count = next(counts[spec["key"]] for spec in config["facilities"] if spec["kind"] == "road")
    sources = "、".join(f"{counts[spec['key']]} 个 {spec['label']}"
                       for spec in config["facilities"] if spec["kind"] == "building")
    style_run(p.add_run(
        f"共 {len(maps)} 幅局部图，覆盖 {road_count} 个道路与步道要素的全部相交段"
        f"（{km:.2f} km）" + (f"，以及 {sources} 建筑轮廓。" if sources else "。")
        + "索引框编号对应后续局部图。"
    ), size=10.0)
    picture(doc, directory/"index.jpg", 14.65, f"{config['place']}道路与建筑潜在暴露局部图索引。")
    legend(doc, config["facilities"])
    p = compact(doc.add_paragraph(), line=13.8, before=3)
    style_run(p.add_run(
        f"局部图以 {config['observations'][-1]['date']} 后期 SAR 为底图，影像缺测处显示 DEM 晕渲。"
        "图内高亮为该幅编排的相交段与建筑轮廓，短路段和小建筑另加定位符号。"
        "相交位置用于安排现场核查，桥面通行、建筑受淹和损毁情况需另行核验。"
    ), size=9.4)
    p = compact(doc.add_paragraph(), line=13.0, before=3)
    style_run(p.add_run(config["sources"]["facilities"] + "  ｜  " + config["crs"]),
              size=8.7, color="4B616B")
    for start in range(0, len(maps), 3):
        subset = maps[start:start+3]
        first, last = subset[0]["number"], subset[-1]["number"]
        page_title = f"潜在暴露详图  A{first:02d}–A{last:02d}"
        heading(doc, page_title, top=True)
        for detail in subset:
            title = compact(doc.add_paragraph(), line=15.5, after=.3, keep=True)
            number, towns = detail["number"], detail["towns"]
            style_run(title.add_run(f"A{number:02d}  {towns}"), size=10.0, bold=True)
            stats = []
            if detail["road_parts"]:
                meters = detail["road_m"]
                stats.append("道路 " + (f"{meters/1000:.2f} km" if meters >= 1000 else f"{meters:.1f} m"))
            buildings = detail.get("buildings") or {"CBMF v2": detail["cbmf"], "OSM": detail["osm"]}
            stats.extend(f"{source} 建筑 {count} 个" for source, count in buildings.items() if count)
            style_run(title.add_run("   ｜   "+" · ".join(stats)), size=9.0, color="4B616B")
            picture(doc, directory/detail["file"], 16.55, f"A{number:02d} {towns}潜在暴露详图。")
    output.parent.mkdir(exist_ok=True, parents=True)
    doc.save(output)


def write_brief(config, stats, towns, exposure, main_map, output):
    """按已核准的统计和主图生成一页 A4 竖向灾情简报。"""
    settings = config.get("products", {})
    title = settings.get("brief_title", f"{config['place']}洪涝遥感灾情简报")
    doc = Document()
    doc.core_properties.title = title
    properties = doc.styles["Title"].element.find(qn("w:pPr"))
    if properties is not None:
        border = properties.find(qn("w:pBdr"))
        if border is not None:
            properties.remove(border)
    page = doc.sections[0]
    page.page_width, page.page_height = Cm(21), Cm(29.7)
    page.top_margin, page.bottom_margin = Cm(1.55), Cm(1.55)
    page.left_margin, page.right_margin = Cm(1.95), Cm(1.95)
    para = compact(doc.add_paragraph(title, style="Title"), line=30, after=5)
    style_run(para.runs[0], size=23, bold=True, color="000000")
    dates = settings.get("observation_caption") or config["main_dates"]
    para = compact(doc.add_paragraph(), line=16, after=8)
    style_run(para.add_run(dates), size=10, color="4B616B")
    para = compact(doc.add_paragraph(), line=16.5, after=6)
    common = stats["common_km2"]
    coverage = 100*common/stats["roi_km2"]
    after = stats["post_km2"]
    before = stats["pre_km2"]
    lead = (f"共同有效观测区为 {common:,.2f} km²（{coverage:.2f}%）。"
            f"水面面积由前期 {before:,.2f} km² 变为后期 {after:,.2f} km²。"
            if common else "两期影像在案例区内没有共同有效观测，无法比较水面变化。")
    style_run(para.add_run(lead), size=10.6)
    table = doc.add_table(rows=2, cols=3)
    table.autofit = False
    metrics = [
        ("新增水面", f"{stats['gain_km2']:,.2f} km²"),
        ("减少水面", f"{stats['loss_km2']:,.2f} km²"),
        ("水面净变化", f"{stats['net_km2']:+,.2f} km²"),
    ]
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border = OxmlElement(f"w:{edge}")
        border.set(qn("w:val"), "single")
        border.set(qn("w:sz"), "5")
        border.set(qn("w:color"), "D9D9D9")
        borders.append(border)
    table._tbl.tblPr.append(borders)
    for col, (label, value) in enumerate(metrics):
        table.columns[col].width = Cm(5.7)
        for row, (content, background, color, size) in enumerate([
            (label, "233F50", "FFFFFF", 10), (value if common else "—", "EDF3F5", "173849", 13),
        ]):
            cell = table.cell(row, col)
            cell.width = Cm(5.7)
            shading = OxmlElement("w:shd")
            shading.set(qn("w:fill"), background)
            cell._tc.get_or_add_tcPr().append(shading)
            para = compact(cell.paragraphs[0], line=18, before=2, after=2, keep=row == 0)
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            style_run(para.add_run(content), size=size, color=color, bold=True)
    picture(doc, main_map, 17.05, f"{config['place']}两期水面变化图")

    if towns and common:
        heading_line = compact(doc.add_paragraph(style="Heading 2"), line=16, before=3, after=2, keep=True)
        style_run(heading_line.add_run("优先核查区域"), size=12, color="000000", bold=True)
        top = sorted(towns, key=lambda item: float(item["gain_km2"]), reverse=True)
        top = [row for row in top if float(row["gain_km2"]) > 0][:4]
        if top:
            ranked = "、".join(f"{row['town']}（{float(row['gain_km2']):.2f} km²）" for row in top)
            para = compact(doc.add_paragraph(), line=15.8, after=1)
            style_run(para.add_run(f"新增水面主要分布在{ranked}等地。"), size=10.3)
    if exposure is not None and common:
        buildings = "、".join(f"{source} {entry['features']} 个"
                             for source, entry in exposure["buildings"].items())
        para = compact(doc.add_paragraph(), line=15.8, before=2, after=1)
        style_run(para.add_run(f"需核查与新增水面相交的道路和步道 {exposure['road_overlap_km']:.2f} km"
                                + (f"，以及建筑轮廓（{buildings}）。" if buildings else "。")), size=10.3)
    para = compact(doc.add_paragraph(), line=15.3, before=3)
    note = settings.get("brief_note", "变化统计限于两期共同有效观测区。设施叠置用于定位待核查对象，"
                        "实际受淹和通行状态以现场核查为准。")
    style_run(para.add_run(note), size=9.5, color="465D68")
    doc.save(output)
