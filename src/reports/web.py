"""Package a hostable dashboard from the case's reviewed statistics and maps."""

import json
from pathlib import Path
from shutil import copyfile
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image
from pyproj import CRS
from shapely import union_all

from generate import GAIN, LOSS, MISSING, STABLE, load_vector
from map_boundaries import parts


STAT_FIELDS = {
    "roi": "roi_km2", "common": "common_km2", "pre": "pre_km2", "post": "post_km2",
    "gain": "gain_km2", "loss": "loss_km2", "stable": "stable_water_km2", "net": "net_km2",
}
WATER_CLASSES = {"stable_water": "stable", "water_gain": "gain", "water_loss": "loss"}
TEMPLATE = Path(__file__).resolve().parent / "templates" / "web"


def projected_path(geometry, origin):
    """转为以左上角为原点的 SVG 路径，坐标保留 0.1 m。"""
    x0, y0 = origin
    path = []
    for polygon in parts(geometry, "Polygon"):
        # 简化仅用于地图显示，所有统计始终读取已核准的数值。
        simple = polygon.simplify(8, preserve_topology=True)
        if simple.is_empty or abs(simple.area - polygon.area) > .05*polygon.area:
            simple = polygon
        for ring in (simple.exterior, *simple.interiors):
            path.append("M" + " ".join(f"{x-x0:.1f},{y0-y:.1f}" for x, y in list(ring.coords)[:-1]) + "Z")
    return "".join(path)


def write_dashboard(config, statistics, towns, exposure, input_maps, output):
    settings = config["products"]
    web = settings["dashboard"]
    output.mkdir(parents=True, exist_ok=True)
    assets = output / "assets"
    assets.mkdir(exist_ok=True)
    inputs = config["vectors"]
    crs = CRS.from_user_input(config["crs"])
    roi = union_all([geom for geom, _ in load_vector(inputs["roi"], crs)])
    left, bottom, right, top = roi.bounds
    origin = left, top
    geometries = {}
    for geom, attrs in load_vector(inputs["towns"], crs):
        geometries.setdefault(attrs["label"], []).append(geom)
    if {row["town"] for row in towns} != geometries.keys() or len(towns) != len(geometries):
        raise ValueError("乡镇统计表与行政区图层未逐一匹配。")
    districts = []
    for row in towns:
        region = union_all(geometries[row["town"]])
        point = region.representative_point()
        districts.append({
            "name": row["town"], **{key: float(row[column]) for key, column in STAT_FIELDS.items()},
            "path": projected_path(region, origin), "label": [point.x-left, top-point.y],
        })
    water = {name: [] for name in (*WATER_CLASSES.values(), "common")}
    for geom, attrs in load_vector(inputs["changes"], crs):
        if kind := WATER_CLASSES.get(attrs["class"]):
            water[kind].append(projected_path(geom, origin))
    water["common"] = [projected_path(geom, origin) for geom, _ in load_vector(inputs["common"], crs)]
    data = {
        "caseInfo": {
            "place": config["place"], "title": settings.get("brief_title", f"{config['place']}洪涝遥感应急简报"),
            "sensor": settings.get("sensor_label", "SAR"), "dates": settings["web_dates"],
            "priority": config.get("town_labels", []),
            "opticalLabel": config.get("boundaries", {}).get("detail_optical", {}).get("label", "局部参考底图"),
            "crs": config["crs"],
            "sources": settings.get("web_sources", "数据来源 · SAR 精修水面、设施底图及辅助影像。"),
        },
        "extent": [right-left, top-bottom], "roi": projected_path(roi, origin), "towns": districts,
        "layers": {key: "".join(paths) for key, paths in water.items()},
        "colors": {"stable": STABLE, "gain": GAIN, "loss": LOSS, "missing": MISSING, "common": "#EDF0EA"},
        "totals": {key: statistics[column] for key, column in STAT_FIELDS.items()},
        "exposure": exposure,
        "scenes": [len(item.get("files", [])) for item in config["observations"]],
    }
    if web.get("aux_stats"):
        correction = json.loads(Path(web["aux_stats"]).read_text(encoding="utf-8"))
        values = correction["reservoir_corrected"]
        data["s1"] = {"pre": values["pre_km2"], "post": values["post_km2"],
                      "net": values["main_component_net_km2"]}
        data["s1Patch"] = correction["patch"]["area_km2"]
        data["auxDates"] = config.get("aux_dates", "")
    else:
        data["s1"] = None
    for key, spec in web["figures"].items():
        image_path = Path(spec["path"]) if "path" in spec else input_maps / spec["map"]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            crop = spec.get("crop")
            if image.height > image.width:
                crop = spec.get("crop_portrait", crop)
            if crop:
                image = image.crop(tuple(round(value*size) for value, size in zip(crop, image.size*2)))
            image.thumbnail((1700, 1700), Image.Resampling.LANCZOS)
            image.save(assets / f"{key}.webp", quality=91, method=6)
    data["figures"] = list(web["figures"])
    if "change" not in data["figures"]:
        raise ValueError("dashboard.figures 必须包含 change 主图。")
    data["briefNote"] = settings.get("brief_note", "")
    for name in ("index.html", "styles.css", "dashboard.js"):
        copyfile(TEMPLATE / name, output / name)
    (assets / "data.js").write_text(
        "window.FLOOD_BRIEF = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    with ZipFile(output.parent/"web_brief.zip", "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output))
