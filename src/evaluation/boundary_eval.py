"""真值水体内外缓冲带评价：严格像素计数 + 可复用固定点，不是容差 Boundary F1。"""
import csv
import hashlib
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from evaluation.metrics import metrics_from_counts


def boundary_regions(label, valid, radii=(1, 3, 5)):
    """按像素中心欧氏距离生成内侧带、外侧带与核心；剔除邻域不完整的位置。"""
    valid = np.asarray(valid, dtype=bool) & np.isin(label, (0, 1))
    water, land = valid & (label == 1), valid & (label == 0)
    # 必须显式处理空集合，不能让 EDT 把影像外的隐含背景当作水体/岸线。
    to_water = distance_transform_edt(~water) if water.any() else np.full(valid.shape, np.inf)
    to_land = distance_transform_edt(~land) if land.any() else np.full(valid.shape, np.inf)
    # 外围补一圈无效：边缘、NoData 洞及其 d 像素邻域均不参加边界计量。
    to_invalid = distance_transform_edt(np.pad(valid, 1))[1:-1, 1:-1]
    regions = {}
    for radius in radii:
        safe = valid & (to_invalid > radius)
        regions[radius] = dict(
            inner=safe & water & (to_land <= radius),
            outer=safe & land & (to_water <= radius),
            core=safe & water & (to_land > radius),
        )
    return regions


def sample_region(mask, name, radius, region, seed=42, max_points=256):
    """区域内等概率、无放回抽样；只依赖名称/标签区域/种子，与预测和遍历顺序无关。"""
    key = f"{seed}|{Path(name).as_posix()}|{radius}|{region}".encode("utf-8")
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "little"))
    indices = np.flatnonzero(mask)
    selected = np.sort(rng.choice(indices, min(max_points, len(indices)), replace=False))
    return np.column_stack(np.unravel_index(selected, mask.shape))


def confusion_counts(label, prediction, mask):
    """与训练一致，返回 TN、FP、FN、TP；空区域自然返回四个零。"""
    codes = np.asarray(label)[mask].astype(np.int64) * 2 + np.asarray(prediction)[mask].astype(np.int64)
    return np.bincount(codes, minlength=4)


def region_metrics(counts):
    """空区域全部精度记 N/A；外侧只解读 FPR，内侧/核心只解读 Recall。"""
    values = metrics_from_counts(counts)
    if not sum(counts):
        for key in values:
            if key not in ("tn", "fp", "fn", "tp", "valid_pixels"):
                values[key] = float("nan")
    return values


class BoundaryEvaluator:
    """逐图累计评价并流式写出点位，避免把整套预测/百万采样点放在内存中。"""

    def __init__(self, output_dir, radii=(1, 3, 5), max_points=256, seed=42):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.radii, self.max_points, self.seed = tuple(radii), max_points, seed
        if not self.radii or any(r <= 0 for r in self.radii) or max_points < 0:
            raise ValueError("缓冲半径须为正数，采样点数量不能为负数")
        self.rows, self.totals = [], {}
        self.point_stream = None
        self.methods = None

    def update(self, name, label, valid, predictions, probabilities=None, thresholds=None):
        """所有方法必须在同一网格上；返回区域与点位，供质量叠加图复用。"""
        label, valid = np.asarray(label), np.asarray(valid, dtype=bool)
        valid = valid & np.isin(label, (0, 1))
        if any(np.shape(p) != label.shape for p in predictions.values()):
            raise ValueError("预测与标签网格尺寸必须一致")
        if self.methods is None:
            self.methods = tuple(predictions)
            self.point_stream = (self.output_dir / "points.csv").open("w", newline="", encoding="utf-8-sig")
            fields = ["name", "label_sha256", "valid_sha256", "shape", "radius_px", "region", "row", "col", "gt", "seed", "valid_rule"]
            for method in self.methods:
                fields.extend((f"{method}_prediction", f"{method}_probability", f"{method}_threshold"))
            self.point_writer = csv.DictWriter(self.point_stream, fields)
            self.point_writer.writeheader()
        if tuple(predictions) != self.methods:
            raise ValueError("同一次评价的方法名单及顺序须固定")
        regions = boundary_regions(label, valid, self.radii)
        digest = hashlib.sha256(np.where(valid, label, -1).astype("<i2").tobytes()).hexdigest()
        valid_digest = hashlib.sha256(valid.tobytes()).hexdigest()
        point_sets = {}
        for radius, areas in regions.items():
            points = {region: sample_region(mask, name, radius, region, self.seed, self.max_points)
                      for region, mask in areas.items()}
            point_sets[radius] = points
            for region, coords in points.items():
                for row, col in coords:
                    record = dict(name=name, label_sha256=digest, valid_sha256=valid_digest,
                                  shape=f"{label.shape[0]}x{label.shape[1]}", radius_px=radius,
                                  region=region, row=int(row), col=int(col), gt=int(label[row, col]),
                                  seed=self.seed, valid_rule="valid 0/1 label; full Euclidean radius neighborhood")
                    for method, prediction in predictions.items():
                        probability = (probabilities or {}).get(method)
                        record.update({f"{method}_prediction": int(prediction[row, col]),
                                       f"{method}_probability": "" if probability is None else float(probability[row, col]),
                                       f"{method}_threshold": (thresholds or {}).get(method, "")})
                    self.point_writer.writerow(record)
            # boundary 是 inner+outer；不含 core。抽样两侧定额会改变类比例，单独标识 scope。
            for method, prediction in predictions.items():
                for scope in ("all_pixels", "sample_points"):
                    counts_by_region = {}
                    for region, mask in areas.items():
                        if scope == "all_pixels":
                            counts = confusion_counts(label, prediction, mask)
                        else:
                            rc = points[region]
                            truth = label[rc[:, 0], rc[:, 1]]
                            pred = prediction[rc[:, 0], rc[:, 1]]
                            counts = confusion_counts(truth, pred, np.ones(len(rc), dtype=bool))
                        counts_by_region[region] = counts
                    counts_by_region["boundary"] = counts_by_region["inner"] + counts_by_region["outer"]
                    for region, counts in counts_by_region.items():
                        key = (method, radius, region, scope)
                        self.totals[key] = self.totals.get(key, np.zeros(4, dtype=np.int64)) + counts
                        self.rows.append(dict(name=name, method=method, radius_px=radius, region=region,
                                              scope=scope, **region_metrics(counts)))
        return regions, point_sets

    def save(self):
        """输出全区域汇总与逐图平均；NaN 不参与平均，并报告有定义的图像数量。"""
        if self.point_stream is not None:
            self.point_stream.close()
        if not self.rows:
            return []
        summary = []
        for key, counts in self.totals.items():
            method, radius, region, scope = key
            row = dict(method=method, radius_px=radius, region=region, scope=scope, **region_metrics(counts))
            subset = [r for r in self.rows if (r["method"], r["radius_px"], r["region"], r["scope"]) == key]
            row["images_with_region"] = sum(r["valid_pixels"] > 0 for r in subset)
            for metric in ("f1", "recall", "false_positive_rate"):
                values = [r[metric] for r in subset if np.isfinite(r[metric])]
                row[f"macro_{metric}"] = float(np.mean(values)) if values else float("nan")
                row[f"macro_{metric}_images"] = len(values)
            summary.append(row)
        for filename, rows in (("per_image.csv", self.rows), ("metrics.csv", summary)):
            with (self.output_dir / filename).open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        return summary

    def close(self):
        """异常中止时也可关闭点位文件。"""
        if self.point_stream is not None:
            self.point_stream.close()


def save_boundary_figure(label, valid, regions, points, output_path):
    """保存固定审查位置：红=外侧，黄=内侧，蓝=核心；叉号是对应像素中心。"""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    figure = Figure(figsize=(5 * len(regions), 5))
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, len(regions), squeeze=False)[0]
    colors = dict(inner=(1., .8, .1), outer=(1., .2, .2), core=(.2, .5, 1.))
    for axis, (radius, areas) in zip(axes, regions.items()):
        rgb = np.full((*label.shape, 3), .12)
        rgb[~valid] = 0
        for region, mask in areas.items():
            rgb[mask] = colors[region]
        axis.imshow(rgb)
        for region, coords in points[radius].items():
            axis.scatter(coords[:, 1], coords[:, 0], s=5, c="white", marker="+", linewidths=.3)
        axis.set_title(f"d={radius}px | red outer / yellow inner / blue core")
        axis.axis("off")
    figure.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=120, bbox_inches="tight", pad_inches=.2)
