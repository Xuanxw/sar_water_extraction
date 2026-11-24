"""Generate a complete facility detail atlas from the reviewed exposure layers."""
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
import json
from math import isclose
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, patheffects
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
import numpy as np
from pyproj import CRS
from shapely import STRtree, union_all
from shapely.geometry import box

from generate import (BRIDGE, BUILDINGS, INK, MISSING, ROAD, RULE, WATER,
                      backdrop, bbox, fill, frame, lines, load_vector,
                      raster_view, terrain_view)
from map_boundaries import parts
from place_labels import label_map, read_labels

# The wide frame leaves room for three readable maps on a portrait A4 page.
SPAN_X, SPAN_Y = 5100, 2350
RATIO = SPAN_X / SPAN_Y
MARGIN = 105


@dataclass
class Feature:
    category: str
    kind: str
    geometry: object
    attrs: dict

    @property
    def identity(self):
        return (self.category, self.attrs["source_id"])


def partition(features):
    """Assign each feature geometry to a frame which contains it in full."""
    if not features:
        return []
    bounds = np.asarray([feature.geometry.bounds for feature in features])
    horizontal = bounds[:, 2] - SPAN_X + MARGIN
    vertical = bounds[:, 3] - SPAN_Y + MARGIN
    x_candidates = [set(np.flatnonzero((bounds[:, 0] >= x+MARGIN-1e-8) &
                                       (bounds[:, 2] <= x+SPAN_X-MARGIN+1e-8)))
                    for x in np.unique(horizontal)]
    y_candidates = [set(np.flatnonzero((bounds[:, 1] >= y+MARGIN-1e-8) &
                                       (bounds[:, 3] <= y+SPAN_Y-MARGIN+1e-8)))
                    for y in np.unique(vertical)]
    candidates = {frozenset(a & b) for a, b in product(x_candidates, y_candidates) if a & b}
    candidates = sorted(candidates, key=lambda cell: (-len(cell), tuple(sorted(cell))))
    remaining = set(range(len(features)))
    frames = []
    while remaining:
        best = max(candidates, key=lambda cell: len(cell & remaining)) & remaining
        if not best:
            raise ValueError("超出最大图幅的设施，请核对图幅尺寸。")
        remaining.difference_update(best)
        indexes = sorted(best)
        x0, y0 = bounds[indexes, :2].min(axis=0)
        x1, y1 = bounds[indexes, 2:].max(axis=0)
        width = max(1180, x1-x0+2*MARGIN, (y1-y0+2*MARGIN)*RATIO)
        height = width/RATIO
        x, y = (x0+x1)/2, (y0+y1)/2
        extent = box(x-width/2, y-height/2, x+width/2, y+height/2)
        if not all(extent.covers(features[index].geometry) for index in indexes):
            raise ValueError("局部图边缘截断设施。")
        frames.append((extent, indexes))
    return sorted(frames, key=lambda item: (-round(item[0].centroid.y/2300), item[0].centroid.x))


def figure(bounds, inches=(7.2, 3.34)):
    fig = plt.figure(figsize=inches, dpi=225, facecolor="white")
    ax = fig.add_axes([.0015, .005, .997, .990])
    frame(ax, bounds, north=True, small=True)
    return fig, ax


def detail_figure(bounds):
    """Place orientation and the scale below the image, away from facilities."""
    fig = plt.figure(figsize=(7.2, 3.34), dpi=225, facecolor="white")
    ax = fig.add_axes([.002, .075, .996, .92])
    x0, y0, x1, y1 = bounds
    ax.set(xlim=(x0, x1), ylim=(y0, y1), xticks=[], yticks=[], facecolor="#EBECE5")
    ax.set_aspect("equal")
    for edge in ax.spines.values():
        edge.set(color=RULE, linewidth=.55)
    ax.apply_aspect()
    slot = ax.get_position()
    bar = .04 + slot.x0
    length = max(value for value in (100, 200, 250, 500, 1000) if value <= .23*(x1-x0))
    endpoint = bar + slot.width*length/(x1-x0)
    fig.add_artist(plt.Line2D([bar, endpoint], [.037, .037], transform=fig.transFigure, color=INK, lw=1.35))
    fig.text(bar-.009, .038, "0", ha="right", va="center", fontsize=7.6, color=INK)
    fig.text(endpoint+.009, .039, f"{length:g} m" if length < 1000 else "1 km",
             ha="left", va="center", fontsize=7.6, color=INK)
    fig.text(slot.x1-.02, .037, "↑ N", ha="right", va="center", fontsize=8.0, color=INK)
    return fig, ax


def save(fig, path):
    from io import BytesIO
    from PIL import Image
    png = BytesIO()
    fig.savefig(png, format="png", dpi=225)
    plt.close(fig)
    png.seek(0)
    with Image.open(png) as image:
        image.convert("RGB").save(path, "JPEG", quality=91, optimize=True, subsampling=0)


def road_label_names(features):
    return tuple(dict.fromkeys(f.attrs["name"] for f in features if f.attrs.get("name")))


def draw_detail(output, config, extent, selected, towns, common, changes, index, crs):
    bounds = extent.bounds
    fig, ax = detail_figure(bounds)
    bottom = terrain_view(config["dem_files"], crs, bounds, size=1450)
    post = config["observations"][-1]
    sar = raster_view(post["files"], crs, bounds, size=1450,
                      units=config.get("observation_units", "power"))
    # DEM fills missing SAR pixels. The blended layer keeps fine SAR texture legible.
    display = np.where(np.isfinite(sar), .15 + sar*.65, bottom)
    backdrop(ax, display, bounds)
    fill(ax, [extent.difference(common)], MISSING, alpha=.19, z=1)
    nearby = [(changes[i][0].intersection(extent), changes[i][1]["class"])
              for i in index.query(extent, predicate="intersects")]
    for key, color, _ in WATER:
        is_gain = key == "water_gain"
        fill(ax, [g for g, kind in nearby if kind == key], color,
             edge="#FFE1A0" if is_gain else "none", width=.24,
             alpha=.77 if is_gain else .32, z=3 if is_gain else 2)
    lines(ax, [g.boundary.intersection(extent) for g, _ in towns if g.intersects(extent)],
          "#E2DAC3", .55, 4, "--")
    for flag, color in [(0, ROAD), (1, BRIDGE)]:
        paths = [f.geometry for f in selected if f.kind == "road"
                 and f.attrs["bridge_tunnel"] == flag]
        lines(ax, paths, "#FFF5C9", 2.35, 20)
        lines(ax, paths, color, 1.35, 21, "--" if flag else "-")
        # A short intersecting segment can be smaller than the stroke on paper.
        points = [g.interpolate(.5, normalized=True) for g in paths if g.length < 18]
        ax.scatter([p.x for p in points], [p.y for p in points], color=color,
                   s=11, edgecolor="#FFF5C9", linewidth=.55, zorder=21)
    groups = defaultdict(list)
    building_keys = [spec["key"] for spec in config["facilities"] if spec["kind"] == "building"]
    for feature in selected:
        if feature.kind == "road":
            continue
        if feature.category == building_keys[0]:
            color = BUILDINGS[0 if feature.attrs["overlap_pct"] >= 50 else 1]
        else:
            color = BUILDINGS[2]
        groups[color].append(feature.geometry)
    for color, geometries in groups.items():
        fill(ax, geometries, color, edge="white", width=.75, z=22)
        # Retain the measured footprint and use a ring to locate sub-symbol polygons.
        points = [g.representative_point() for g in geometries if g.area < 175]
        ax.scatter([p.x for p in points], [p.y for p in points], facecolors="none", edgecolors="white",
                   s=40, linewidth=1, zorder=23)
        ax.scatter([p.x for p in points], [p.y for p in points], facecolors="none", edgecolors=color,
                   s=40, linewidth=.65, zorder=24)
    labels = read_labels(config["labels"]["sources"], crs, bounds)
    label_map(ax, labels, bounds, 10, small=True, priority=road_label_names(selected))
    save(fig, output)


def draw_index(path, config, roi, towns, changes, frames, crs):
    full = bbox(roi.bounds, 1.09)
    fig, ax = figure(full, (6.8, 6.8))
    backdrop(ax, terrain_view(config["dem_files"], crs, full, size=1280), full)
    for key, color, _ in WATER:
        fill(ax, [g for g, p in changes if p["class"] == key], color, alpha=.65)
    fill(ax, [g for g, _ in towns], "none", edge="#9DAEAE", width=.45, z=3)
    fill(ax, [roi], "none", edge=INK, width=.8, z=4)
    ax.apply_aspect()
    view = ax.get_window_extent()
    marks = []
    for number, (extent, _) in enumerate(frames, 1):
        x0, y0, x1, y1 = extent.bounds
        ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fc="none", ec="#AB4917", lw=.75, zorder=6))
        anchor = extent.centroid.coords[0]
        x, y = ax.transData.transform(anchor)
        marks.append((number, anchor, x, y))
    occupied = []
    pixels_per_point = fig.dpi/72
    for n, anchor, x, y in marks:
        for radius in (0, 10, 18, 26, 36, 49, 62, 80):
            directions = [0] if radius == 0 else np.arange(0, 2*np.pi, np.pi/4)
            found = False
            for angle in directions:
                ox, oy = radius*np.cos(angle), radius*np.sin(angle)
                px = x+ox*pixels_per_point
                py = y+oy*pixels_per_point
                text_box = Bbox.from_bounds(px-8*pixels_per_point, py-6*pixels_per_point,
                                             16*pixels_per_point, 12*pixels_per_point)
                if (not view.contains(*text_box.min) or not view.contains(*text_box.max)
                        or any(text_box.overlaps(used) for used in occupied)):
                    continue
                ax.annotate(f"{n:02d}", anchor, xytext=(ox, oy), textcoords="offset points",
                            ha="center", va="center", color="white", fontsize=7.7, weight="bold",
                            bbox={"boxstyle": "round,pad=.18", "fc": "#87360E", "ec": "white", "lw": .68},
                            arrowprops={"arrowstyle": "-", "color": "#864027", "lw": .65} if radius else None,
                            zorder=18)
                occupied.append(text_box)
                found = True
                break
            if found:
                break
        else:
            raise ValueError(f"索引图标号 {n:02d} 无可用位置。")
    for name in config.get("town_labels", ()):
        for geom, props in towns:
            if props.get("label") != name:
                continue
            point = geom.representative_point()
            ax.annotate(name, (point.x, point.y), fontsize=7.5, ha="center", va="center",
                        color=INK, path_effects=[patheffects.withStroke(linewidth=2.0, foreground="white")],
                        zorder=12, clip_on=True)
    save(fig, path)


def atlas_features(config, facilities, crs):
    """读取已发布的暴露图层。道路按相交线分量拆分，建筑保留完整轮廓。"""
    features = []
    counts = {}
    for spec in config["facilities"]:
        category = spec["key"]
        entries = load_vector({"path": facilities, "layer": category}, crs)
        counts[category] = len(entries)
        for geometry, attrs in entries:
            shapes = list(parts(geometry, "LineString")) if spec["kind"] == "road" else [geometry]
            if spec["kind"] == "road" and not isclose(
                sum(line.length for line in shapes), attrs["overlap_m"], abs_tol=1e-6
            ):
                raise ValueError(f"道路几何与统计长度不一致 {attrs['source_id']}")
            features.extend(Feature(category, spec["kind"], shape, attrs) for shape in shapes)
    return features, counts


def draw_atlas(config, facilities, output):
    """以案例米制 CRS 生成索引与局部图，返回供 Word 和核查清单复用的图幅清单。"""
    output.mkdir(parents=True, exist_ok=True)
    crs = CRS.from_user_input(config["crs"])
    if font := config.get("font_file"):
        font_manager.fontManager.addfont(font)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    source = config["vectors"]
    roi = union_all([g for g, _ in load_vector(source["roi"], crs)])
    towns = load_vector(source["towns"], crs)
    common = union_all([g for g, _ in load_vector(source["common"], crs)])
    changes = load_vector(source["changes"], crs)
    index = STRtree([g for g, _ in changes])
    features, expected = atlas_features(config, facilities, crs)

    frames = partition(features)
    assignments = [i for _, indices in frames for i in indices]
    if sorted(assignments) != list(range(len(features))):
        raise ValueError("设施几何图幅覆盖不完整或重复分配。")
    if frames:
        draw_index(output/"index.jpg", config, roi, towns, changes, frames, crs)
    metadata = []
    for number, (extent, indexes) in enumerate(frames, 1):
        selected = [features[i] for i in indexes]
        areas = defaultdict(float)
        for feature in selected:
            for boundary, props in towns:
                if boundary.covers(feature.geometry.representative_point()):
                    name = props["label"]
                    areas[name] += feature.geometry.length if feature.kind == "road" else 20
                    break
        label = "、".join(sorted(areas, key=areas.get, reverse=True)) or config["place"]
        filename = f"detail_{number:02d}.jpg"
        draw_detail(output/filename, config, extent, selected, towns, common, changes, index, crs)
        metadata.append({"file": filename, "number": number, "towns": label,
                         "bounds": extent.bounds,
                         "width_m": extent.bounds[2]-extent.bounds[0],
                         "road_parts": sum(f.kind == "road" for f in selected),
                         "road_m": sum(f.geometry.length for f in selected if f.kind == "road"),
                         "buildings": {spec["label"]: sum(f.category == spec["key"] for f in selected)
                                       for spec in config["facilities"] if spec["kind"] == "building"}})
        print(f"{number:02d}/{len(frames)} {label}", flush=True)

    counts = {name: len({f.identity for f in features if f.category == name}) for name in expected}
    assert counts == expected
    stats = (facilities.parent/"exposure_summary.json")
    if stats.exists():
        source = json.loads(stats.read_text(encoding="utf-8"))
        assert isclose(sum(d["road_m"] for d in metadata)/1000, source["road_overlap_km"], abs_tol=1e-7)
        for spec in config["facilities"]:
            count = (source["road_features"] if spec["kind"] == "road"
                     else source["buildings"][spec["label"]]["features"])
            assert counts[spec["key"]] == count
    manifest = {"maps": metadata, "counts": counts}
    (output/"manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest
