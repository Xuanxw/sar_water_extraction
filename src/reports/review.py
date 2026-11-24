"""Build review layers and a filterable workbook from the atlas feature assignment."""

from collections import Counter, defaultdict
from math import isclose

import fiona
from pyproj import CRS
from shapely import STRtree, union_all
from shapely.geometry import MultiLineString, mapping

from atlas import atlas_features, partition
from generate import load_vector
from map_boundaries import parts


REVIEW_FIELDS = {"review_id": "str", "map_no": "str", "town": "str", "report_page": "int"}
EDIT_FIELDS = {"review_status": "str", "remarks": "str"}
ROAD_FIELDS = {
    **REVIEW_FIELDS, "source_id": "str", "road_name": "str", "road_ref": "str", "road_class": "str",
    "overlap_m": "float", "bridge": "str", "tunnel": "str", "bridge_tunnel": "str", **EDIT_FIELDS,
}
BUILDING_FIELDS = {
    **REVIEW_FIELDS, "source": "str", "outline_id": "str", "footprint_m2": "float",
    "overlap_m2": "float", "overlap_pct": "float", **EDIT_FIELDS,
}
INDEX_FIELDS = {
    "map_no": "str", "report_page": "int", "atlas_towns": "str", "review_towns": "str",
    "road_length_m": "float", "road_rows": "int", "bridge_rows": "int", "buildings": "int",
}


def road_tag(value):
    tag = str(value).strip().lower()
    if tag in {"t", "true", "1", "yes"}:
        return "有标记"
    if tag in {"f", "false", "0", "no"}:
        return "无标记"
    return "未知"


def split_towns(geometry, boundaries, index):
    """逐乡镇切分相交线，边界共享线只分配一次，长度单位为米。"""
    remaining = geometry
    for i in sorted(index.query(geometry, predicate="intersects")):
        boundary, name = boundaries[i]
        if remaining.is_empty:
            break
        lines = list(parts(remaining.intersection(boundary), "LineString"))
        if lines:
            yield name, lines
            remaining = remaining.difference(boundary)
    remainder = list(parts(remaining, "LineString"))
    if sum(line.length for line in remainder) > 1e-6:
        yield "行政界未匹配", remainder


def write_layer(path, layer, kind, fields, records, crs):
    with fiona.open(path, "w", driver="GPKG", layer=layer, crs_wkt=crs.to_wkt(),
                    schema={"geometry": kind, "properties": fields}) as sink:
        sink.writerecords({"geometry": mapping(g), "properties": attrs} for g, attrs in records)


def collect_review(config, facilities, manifest):
    """复用局部图幅分配，验证各图幅道路长度和建筑数量。"""
    crs = CRS.from_user_input(config["crs"])
    by_town = defaultdict(list)
    for geometry, attrs in load_vector(config["vectors"]["towns"], crs):
        by_town[attrs["label"]].append(geometry)
    boundaries = [(union_all(geometry), name) for name, geometry in sorted(by_town.items())]
    index = STRtree([geometry for geometry, _ in boundaries])
    features, _ = atlas_features(config, facilities, crs)
    road_spec = next(spec for spec in config["facilities"] if spec["kind"] == "road")
    with fiona.open(road_spec["path"], layer=road_spec["layer"]) as source:
        streets = {str(f.properties[road_spec["id"]]): dict(f.properties) for f in source}

    reviews = {"map_index": [], "road_review": [], "building_review": []}
    for (extent, indexes), entry in zip(partition(features), manifest["maps"], strict=True):
        if not all(isclose(x, y, abs_tol=1e-7, rel_tol=0)
                   for x, y in zip(extent.bounds, entry["bounds"])):
            raise ValueError("已有图幅和当前设施不匹配，请重新生成详图。")
        number = entry["number"]
        first_page = config.get("products", {}).get("first_detail_page", 6)
        attributes = {"map_no": f"A{number:02d}", "report_page": first_page+(number-1)//3}
        editable = {"review_status": "待核查", "remarks": None}
        roads = defaultdict(list)
        buildings = []
        for i in indexes:
            feature = features[i]
            attrs = feature.attrs
            if feature.kind == "road":
                for town, segments in split_towns(feature.geometry, boundaries, index):
                    roads[town, attrs["source_id"]].extend(segments)
                continue
            point = feature.geometry.representative_point()
            matches = sorted(index.query(point, predicate="covered_by"))
            town = boundaries[matches[0]][1] if matches else "行政界未匹配"
            buildings.append((town, attrs, feature.geometry))
        used_towns = set()
        lengths = []
        bridges = 0
        for order, ((town, source_id), lines) in enumerate(sorted(roads.items()), 1):
            raw = streets[source_id]
            bridge = road_tag(raw.get("bridge"))
            tunnel = road_tag(raw.get("tunnel"))
            flag = "有标记" if "有标记" in (bridge, tunnel) else (
                "未知" if "未知" in (bridge, tunnel) else "无标记")
            line = MultiLineString(lines)
            used_towns.add(town)
            lengths.append(line.length)
            bridges += flag == "有标记"
            reviews["road_review"].append((line, {
                **attributes, **editable, "review_id": f"A{number:02d}-R{order:02d}",
                "town": town, "source_id": source_id, "road_name": raw.get("name") or "底图未命名",
                "road_ref": raw.get("ref") or None, "road_class": raw.get("fclass"),
                "overlap_m": line.length, "bridge": bridge, "tunnel": tunnel, "bridge_tunnel": flag,
            }))

        for order, (town, attrs, geometry) in enumerate(
            sorted(buildings, key=lambda row: (row[0], row[1]["source"], row[1]["source_id"])), 1
        ):
            used_towns.add(town)
            reviews["building_review"].append((geometry, {
                **attributes, **editable, "review_id": f"A{number:02d}-B{order:02d}",
                "town": town, "source": attrs["source"], "outline_id": attrs["source_id"],
                "footprint_m2": geometry.area, "overlap_m2": attrs["overlap_m2"],
                "overlap_pct": attrs["overlap_pct"],
            }))
        expected = entry.get("buildings") or {"CBMF v2": entry.get("cbmf", 0), "OSM": entry.get("osm", 0)}
        building_counts = Counter(attrs["source"] for _, attrs, _ in buildings)
        if not isclose(sum(lengths), entry["road_m"], abs_tol=1e-5) or building_counts != Counter(expected):
            raise ValueError(f"A{number:02d} 落入清单的几何与已制图统计不一致。")
        reviews["map_index"].append((extent, {
            **attributes, "atlas_towns": entry["towns"], "review_towns": "、".join(sorted(used_towns)),
            "road_length_m": sum(lengths), "road_rows": len(roads), "bridge_rows": bridges,
            "buildings": len(buildings),
        }))
    return crs, reviews


def write_workbook(path, reviews):
    """把同一批 GIS 属性写成三张可筛选的 Excel 工作表。"""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.table import Table, TableStyleInfo

    schemas = [
        ("图幅索引", "map_index", "AtlasIndex", {
            "map_no": "图号", "report_page": "报告页码", "atlas_towns": "报告标注乡镇",
            "review_towns": "清单涉及乡镇", "road_length_m": "道路相交长度 (m)",
            "road_rows": "道路记录数", "bridge_rows": "桥隧标记记录数", "buildings": "建筑轮廓数",
        }),
        ("道路核查", "road_review", "RoadReview", {
            "map_no": "图号", "town": "乡镇 / 街道", "review_id": "核查编号", "report_page": "报告页码",
            "road_name": "道路名称", "road_ref": "路线编号", "overlap_m": "相交长度 (m)",
            "bridge": "桥梁", "tunnel": "隧道", "bridge_tunnel": "桥隧标记", "source_id": "道路源 ID",
            "road_class": "道路类型", "review_status": "核查状态", "remarks": "备注",
        }),
        ("建筑核查", "building_review", "BuildingReview", {
            "map_no": "图号", "town": "乡镇 / 街道", "review_id": "核查编号", "report_page": "报告页码",
            "source": "建筑来源", "outline_id": "轮廓 ID", "footprint_m2": "轮廓面积 (m²)",
            "overlap_m2": "重叠面积 (m²)", "overlap_pct": "重叠比例", "review_status": "核查状态", "remarks": "备注",
        }),
    ]
    notes = {
        "map_index": "QGIS 中用 map_no 筛选图号，用 review_id 查核查对象。",
        "road_review": "同一道路可跨图幅或乡镇。相交长度可加总，道路数量按源 ID 去重。",
        "building_review": "建筑保留完整轮廓，重叠比例为新增水面相交面积除以轮廓面积。",
    }
    book = Workbook()
    book.remove(book.active)
    for title, key, table_name, columns in schemas:
        sheet = book.create_sheet(title)
        sheet.sheet_view.showGridLines = False
        sheet.append([title])
        sheet["A1"].font = Font(name="Microsoft YaHei", size=15, color="173849", bold=True)
        sheet.append([notes[key]])
        sheet["A2"].font = Font(name="Microsoft YaHei", color="506370")
        sheet.append(list(columns.values()))
        for _, attrs in reviews[key]:
            sheet.append([attrs[field]/100 if field == "overlap_pct" else attrs[field] for field in columns])
        for col, field in enumerate(columns, 1):
            letter = get_column_letter(col)
            sheet.column_dimensions[letter].width = 34 if field in ("atlas_towns", "review_towns", "remarks") else 19
            for (cell,) in sheet.iter_rows(min_row=4, min_col=col, max_col=col):
                if field in ("road_length_m", "overlap_m", "footprint_m2", "overlap_m2"):
                    cell.number_format = "#,##0.0"
                elif field == "overlap_pct":
                    cell.number_format = "0.00%"
                elif field in EDIT_FIELDS:
                    cell.fill = PatternFill("solid", fgColor="FFF2D9")
            header = sheet.cell(3, col)
            header.font = Font(name="Microsoft YaHei", color="FFFFFF", bold=True)
            header.fill = PatternFill("solid", fgColor="28495E")
            header.alignment = Alignment(vertical="center")
        if reviews[key]:
            table = Table(displayName=table_name, ref=f"A3:{get_column_letter(len(columns))}{sheet.max_row}")
            table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True)
            sheet.add_table(table)
        sheet.freeze_panes = "A4"
    book.save(path)


def write_review(config, facilities, manifest, output):
    title = config.get("products", {}).get("review_title", f"{config['place']}设施潜在暴露核查清单")
    crs, reviews = collect_review(config, facilities, manifest)
    layers = [
        ("map_index", "Polygon", INDEX_FIELDS),
        ("road_review", "MultiLineString", ROAD_FIELDS),
        ("building_review", "MultiPolygon", BUILDING_FIELDS),
    ]
    for layer, geometry, fields in layers:
        write_layer(output/f"{title}.gpkg", layer, geometry, fields, reviews[layer], crs)
    write_workbook(output/f"{title}.xlsx", reviews)
