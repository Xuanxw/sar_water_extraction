"""Build selected flood-response report products in a new output directory."""

import argparse
import csv
import json
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp
from time import perf_counter


PRODUCTS = ("maps", "brief", "atlas", "review", "dashboard")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def product_settings(case, config):
    """解析案例配置中报告输入，所有相对路径都相对于案例 JSON。"""
    settings = config.setdefault("products", {})
    for key in ("maps_dir", "atlas_dir", "facility_gpkg", "facility_summary", "report_docx",
                "statistics", "town_csv"):
        if settings.get(key):
            settings[key] = (case.parent / settings[key]).resolve()
    web = settings.get("dashboard", {})
    if web.get("aux_stats"):
        web["aux_stats"] = (case.parent / web["aux_stats"]).resolve()
    for spec in web.get("figures", {}).values():
        if "path" in spec:
            spec["path"] = (case.parent / spec["path"]).resolve()
    return settings


def load_statistics(settings):
    stats = read_json(settings["statistics"])
    if key := settings.get("statistics_key"):
        stats = stats[key]
    towns = []
    if path := settings.get("town_csv"):
        with path.open(newline="", encoding="utf-8-sig") as handle:
            towns = [row for row in csv.DictReader(handle)
                     if not settings.get("town_sensor") or row["sensor"] == settings["town_sensor"]]
    return stats, towns


def build_products(case, output, selected):
    """独立生成选中的成果，失败时保留输入和已有成果。"""
    if output.exists():
        raise FileExistsError(f"请使用新的成果目录：{output}")
    config = read_json(case)
    if set(selected) - {"brief"}:
        from generate import load_case
        config = load_case(case)
    settings = product_settings(case, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(mkdtemp(prefix=f".{output.name}_", dir=output.parent))
    started = perf_counter()
    try:
        input_maps = settings.get("maps_dir", stage/"maps")
        facilities = settings.get("facility_gpkg", input_maps/"facility_exposure.gpkg")
        summary_file = settings.get("facility_summary", input_maps/"exposure_summary.json")

        if "maps" in selected:
            from generate import draw_case
            print("生成四张专题图", flush=True)
            input_maps = stage / "maps"
            draw_case(config, input_maps)
            facilities = input_maps / "facility_exposure.gpkg"
            summary_file = input_maps / "exposure_summary.json"

        if {"brief", "dashboard"} & set(selected):
            statistics, towns = load_statistics(settings)
            exposure = read_json(summary_file) if summary_file.is_file() else None

        if "brief" in selected:
            from documents import write_brief
            print("生成 Word 灾情简报", flush=True)
            title = settings.get("brief_title", f"{config['place']}洪涝遥感灾情简报")
            write_brief(config, statistics, towns, exposure, input_maps/"01_water_change.png",
                        stage/f"{title}.docx")

        if {"atlas", "review"} & set(selected):
            atlas_dir = settings.get("atlas_dir")
            if "atlas" in selected or atlas_dir is None or "maps" in selected:
                from atlas import draw_atlas
                print("生成设施暴露详图", flush=True)
                atlas_dir = stage / "atlas"
                manifest = draw_atlas(config, facilities, atlas_dir)
            else:
                manifest = read_json(atlas_dir/"manifest.json")

            if "atlas" in selected and settings.get("report_docx"):
                from documents import append_atlas
                template = settings["report_docx"]
                if "maps" in selected:
                    from update_report import update
                    template = stage/"_four_map_report.docx"
                    update(settings["report_docx"], input_maps, template, config.get("report", {}))
                title = settings.get("report_title", f"{config['place']}洪涝SAR应急监测报告_设施暴露详图版")
                append_atlas(template, atlas_dir, stage/f"{title}.docx", config, manifest)
                if "maps" in selected:
                    template.unlink()

            if "review" in selected:
                from review import write_review
                print("生成 QGIS 核查图层与 Excel 清单", flush=True)
                write_review(config, facilities, manifest, stage)

        if "dashboard" in selected:
            from web import write_dashboard
            print("生成网页简报和部署 ZIP", flush=True)
            write_dashboard(config, statistics, towns, exposure, input_maps, stage/"web")

        stage.rename(output)
    finally:
        if stage.exists():
            # stage 由本次运行创建，拒绝清理解析到父目录之外的路径。
            if stage.resolve().parent != output.parent.resolve() or not stage.name.startswith(f".{output.name}_"):
                raise RuntimeError(f"无法确认临时输出路径：{stage}")
            rmtree(stage)
    print(f"成果目录 {output}  |  耗时 {perf_counter()-started:.1f} 秒", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path, help="案例 JSON")
    parser.add_argument("output", type=Path, help="尚不存在的成果目录")
    parser.add_argument("--products", nargs="+", choices=[*PRODUCTS, "all"], default=["brief"])
    args = parser.parse_args()
    selected = PRODUCTS if "all" in args.products else tuple(args.products)
    build_products(args.case.resolve(), args.output.resolve(), selected)
