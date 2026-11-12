"""独立阈值入口；处理参数来自CONFIG_PATH指定的独立YAML。"""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from configuration import load_yaml

# 唯一入口配置路径；数据路径和算法参数全部在YAML中修改。
# 本次运行密云专项；切换通用val时改为configs/threshold_segmentation.yaml。
CONFIG_PATH=Path(r'E:\Code_projects\sar_water_extraction\configs\北京密云区\threshold_segmentation_20261112.yaml')


def main(config_path=None):
    """读取配置并传入现有处理流程；不读取train.yaml或image_process.yaml。"""
    path=Path(config_path) if config_path is not None else CONFIG_PATH
    config=load_yaml(path)
    if config['run']['mode']=='org_scenes':
        from thres_segmentation.org_pipeline import run_org_scenes
        return run_org_scenes(config,config_path=path)
    paths=config['paths'];run=config['run'];inputs=config['input'];output=config['output']
    mode,layout=run['mode'],run['data_layout']
    auxiliary=config['auxiliary'];landcover=config['landcover']
    analysis=dict(config['analysis'])
    for name in ('reference_path','reference_coverage_path','elevation_path'):
        analysis[name]=paths[name.upper()]
    print(f'阈值分割配置：{path}',flush=True)
    if mode=='gui':
        print('旧GUI的算法参数仍由界面设置；统一YAML批处理使用dataset模式。',flush=True)
        from thres_segmentation.gui import run_gui
        return run_gui()
    if mode not in ('dataset','download') or layout not in ('full_scene','patch'):
        raise ValueError('mode为dataset/download/gui；data_layout为full_scene/patch')
    if layout=='full_scene' and (paths['SPLIT_MANIFEST'] is None or landcover['enabled']):
        raise ValueError('BC整景需要SPLIT_MANIFEST，landcover.enabled必须为false')
    scene_analysis=None
    if mode=='download' or auxiliary['download_history'] or auxiliary['download_elevation']:
        from thres_segmentation.auxiliary_data import selected_images,prepare_auxiliary_inputs
        if mode=='download' and not (auxiliary['download_history'] or auxiliary['download_elevation']):
            raise ValueError('download模式需开启至少一种辅助数据下载')
        images=selected_images(paths['IMAGE_DIR'],paths['SPLIT_MANIFEST'],
                               inputs['scene_prefix'] if layout=='full_scene' else None)
        scene_analysis=prepare_auxiliary_inputs(images,paths['AUXILIARY_CACHE_DIR'],
                analysis_options=analysis,require_depth_reference=mode!='download',**auxiliary)
        if mode=='download':return scene_analysis
    output_dir=paths['OUTPUT_DIR'] or paths['OUTPUT_ROOT']/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    print(f"输入：{paths['IMAGE_DIR']}；标签：{paths['LABEL_DIR']}；输出：{output_dir}",flush=True)
    print(f"阈值参数：{config['threshold']}；波段：{inputs['bands']}",flush=True)
    common=dict(image_dir=paths['IMAGE_DIR'],label_dir=paths['LABEL_DIR'],
                split_manifest=paths['SPLIT_MANIFEST'],bands=inputs['bands'],output_dir=output_dir,
                threshold_options=config['threshold'],save_masks=output['save_masks'],
                max_visualizations=output['max_visualizations'],analysis_options=analysis,
                scene_analysis_options=scene_analysis)
    if layout=='full_scene':
        from thres_segmentation.scene_pipeline import run_scene_segmentation
        return run_scene_segmentation(**common,block_size=config['processing']['block_size'],
                cache_dir=paths['CACHE_DIR'],preview_size=output['preview_size'],
                scene_prefix=inputs['scene_prefix'],boundary_radii=output['boundary_radii'],
                quality_review_options=config['quality_review'])
    from thres_segmentation.pipeline import run_segmentation
    return run_segmentation(**common,use_landcover=landcover['enabled'],landcover_band=landcover['band'])


if __name__=='__main__':
    # 可选传入YAML路径；不传则沿用CONFIG_PATH。
    main(sys.argv[1] if len(sys.argv)>1 else None)
