"""整景 SNAP Lee 写出器对照。python scripts/compare_snap_writers.py CONFIG.yaml [实验输出父目录]"""
import csv
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from configuration import load_yaml
from datasets.snap_preprocess import SnapSpeckleFilter, dimap_band_vrt, snap_runtime

SCENE = "BC3-SM-ORG-VV-20240704T150208-008104"
WRITERS = (("bigtiff", "GeoTIFF-BigTIFF", ".tif"), ("dimap", "BEAM-DIMAP", ".dim"))


def scene_input(prepared):
    with (prepared / "split_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        matches = [row for row in csv.DictReader(stream) if row["scene_id"] == SCENE]
    if len(matches) != 1 or int(matches[0]["active"]) != 1:
        raise ValueError(f"需要唯一且启用的场景 {SCENE}，请核对 {prepared}")
    row = matches[0]
    return (prepared / "images" / row["split"] / row["name"]).resolve(strict=True)


def grid(ds):
    return dict(width=ds.width, height=ds.height, bands=ds.count, dtype=ds.dtypes,
                crs=ds.crs.to_wkt() if ds.crs else None, affine=list(ds.transform)[:6], nodata=ds.nodata)


def compare_pixels(left, right, adapted):
    """全景按原生像元比较：共同有效域功率作严格数值比较，掩膜独立核对输入。"""
    with rasterio.open(left) as a, rasterio.open(right) as b, rasterio.open(adapted) as ref:
        shape = a.shape == b.shape == ref.shape and a.count == b.count == ref.count == 1
        crs_equal = a.crs == b.crs == ref.crs and ref.crs is not None
        affine_equal = a.transform == b.transform == ref.transform
        result = dict(grids=[grid(a), grid(b), grid(ref)], same_shape=shape, same_crs=crs_equal,
                      exact_affine=affine_equal, same_dtype=a.dtypes == b.dtypes == ("float32",))
        if not shape:
            return dict(result, equal=False)
        mismatch = valid = differences = left_diff = right_diff = nonfinite = 0
        max_error = 0.
        for _, window in ref.block_windows(1):
            ma = a.read_masks(1, window=window) != 0
            mb = b.read_masks(1, window=window) != 0
            mr = ref.read_masks(1, window=window) != 0
            mismatch += int(np.count_nonzero(ma != mb))
            left_diff += int(np.count_nonzero(ma != mr))
            right_diff += int(np.count_nonzero(mb != mr))
            common = ma & mb
            valid += int(common.sum())
            x = a.read(1, window=window)[common]
            y = b.read(1, window=window)[common]
            differences += int(np.count_nonzero(x != y))
            finite = np.isfinite(x) & np.isfinite(y)
            nonfinite += int(np.count_nonzero(~finite))
            if finite.any():
                max_error = max(max_error, float(np.max(np.abs(x[finite].astype("float64") - y[finite]))))
        result.update(mask_mismatch=mismatch, mask_mismatch_vs_input=[left_diff, right_diff],
                      common_valid=valid, value_mismatch=differences, nonfinite_valid=nonfinite,
                      max_abs_error=max_error if valid > nonfinite else None)
        result["equal"] = bool(crs_equal and affine_equal and result["same_dtype"] and valid
                               and not (mismatch or differences or nonfinite))
        return result


def run_writer(folder, writer, src, lee, config):
    """分别统计输入准备、GPT、恢复与无损压缩，以及生成本地最终文件的墙钟时间（秒）。"""
    tag, format_name, suffix = writer
    folder.mkdir()
    userdir = folder / "snap_userdir"
    userdir.mkdir()
    adapted = folder / "input.tif"
    raw = folder / ("snap_output" + suffix)
    graph = folder / "graph.xml"
    output = folder / "filtered.tif"
    log = folder / "gpt.log"
    start = time.perf_counter()
    print(f"[{tag}] 整理输入 {src.name}", flush=True)
    profile = lee.prepare_input(src, (1,), adapted)
    lee.write_graph(graph, adapted, raw, 1, format_name=format_name)
    params = config["snap"]
    command = [str(config["paths"]["SNAP_GPT_PATH"]), str(graph), "-q", str(params["SNAP_THREADS"]),
               "-c", str(params["SNAP_TILE_CACHE"]), f"-Dsnap.userdir={userdir}"]
    print(f"[{tag}] GPT 开始，日志 {log}", flush=True)
    before = time.perf_counter()
    with log.open("w") as stream:
        proc = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    after = time.perf_counter()
    if proc.returncode:
        raise RuntimeError(f"{format_name} GPT 退出码 {proc.returncode}，见 {log}")
    print(f"[{tag}] GPT 完成 {after-before:.1f} 秒，开始恢复掩膜并压缩", flush=True)
    view = dimap_band_vrt(raw) if tag == "dimap" else raw
    restore_error = None
    try:
        lee.restore_output(src, adapted, view, output, profile, format_name=format_name)
    except ValueError as error:
        # 即使生产网格检查拒绝输出，也保留中间产品，继续完成原生格式诊断。
        restore_error = str(error)
    finish = time.perf_counter()
    times = dict(prepare_s=before-start, gpt_s=after-before, restore_and_compress_s=finish-after,
                 local_end_to_end_s=finish-start)
    print(f"[{tag}] {times}", flush=True)
    return times, (view, output, adapted), restore_error


def save_report(dest, report):
    pending = dest.with_suffix(".tmp")
    pending.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pending.replace(dest)


def main(args):
    if len(args) not in (2, 3):
        raise SystemExit(__doc__)
    config = load_yaml(Path(args[1]))
    paths, processing = config["paths"], config["processing"]
    if (tuple(config["input"]["BANDS"]), processing["FILTER_INPUT_STAGE"], processing["INPUT_DOMAIN"]) != ((1,), "org", "intensity"):
        raise ValueError("此对照需要单波段 BC ORG 线性功率输入")
    src = scene_input(paths["OUTPUT_DIR"])
    parent = Path(args[2]) if len(args) == 3 else paths["SNAP_WORK_DIR"].parent / "snap_writer_bench"
    parent.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="scene362_", dir=parent)).resolve()
    print(f"第 362 景整景对照：{run_dir}", flush=True)
    lee = SnapSpeckleFilter(dict(input_domain="intensity", window_size=processing["FILTER_WINDOW"],
                                 enl=processing["ENL"]))
    report = dict(scene=SCENE, source=str(src), yaml=str(Path(args[1]).resolve()), config=lee.config,
                  threads=config["snap"]["SNAP_THREADS"], tile_cache=config["snap"]["SNAP_TILE_CACHE"],
                  runtime=snap_runtime(str(paths["SNAP_GPT_PATH"])), order=[w[1] for w in WRITERS], times={})
    products = {}
    report["restore_errors"] = {}
    dest = run_dir / "report.json"
    try:
        for writer in WRITERS:
            tag = writer[0]
            report["times"][tag], products[tag], error = run_writer(run_dir / tag, writer, src, lee, config)
            if error:
                report["restore_errors"][tag] = error
            save_report(dest, report)
        a, b = products.values()
        for i, stage in enumerate(("raw", "final")):
            if stage == "final" and report["restore_errors"]:
                report["final"] = dict(equal=False, reason="至少一个中间产品未通过生产网格或数值检查")
                break
            print(f"逐块比较 {stage} 像元、有效域和网格", flush=True)
            report[stage] = compare_pixels(a[i], b[i], a[2])
            print(f"  {stage}: { {k: v for k, v in report[stage].items() if k != 'grids'} }", flush=True)
            save_report(dest, report)
        final = report["final"]
        report["final_matches_with_source_mask"] = final["equal"] and final["mask_mismatch_vs_input"] == [0, 0]
        report["passed"] = report["raw"]["equal"] and report["final_matches_with_source_mask"]
        times = report["times"]
        report["bigtiff_over_dimap_time_ratio"] = {
            k: times["bigtiff"][k] / times["dimap"][k] for k in ("gpt_s", "local_end_to_end_s")
            if k == "gpt_s" or not report["restore_errors"]}
        print(json.dumps({k: report[k] for k in ("bigtiff_over_dimap_time_ratio", "final_matches_with_source_mask", "passed")}, indent=2))
        return 0 if report["passed"] else 1
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        save_report(dest, report)
        print(f"报告：{dest}", flush=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
