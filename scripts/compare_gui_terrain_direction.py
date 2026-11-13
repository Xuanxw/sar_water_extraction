"""重新生成方向修正后的局部地形掩膜，与既有7.5m剖面结果比较；原水体只读。"""
from pathlib import Path
import sys
import json
import shutil
from time import perf_counter
import numpy as np
import rasterio
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from configuration import config_argument
from sar_geometry.main import load_config
from sar_geometry.scene_validation import compare_scene,warped_array
from sar_geometry.terrain_io import sha256,write_json
from thres_segmentation.terrain_geometry import generate_terrain_mask

OUTPUT_DIR=Path(r'E:\Code_projects\sar_water_extraction\codex_analysis\sar_geometry\gui_direction_corrected_025242')


def main(config_path,output=OUTPUT_DIR):
    """冻结原有效域/坡度/入射角，按原SAR网格重算局部掩膜，统计三方共同有效域。"""
    config=load_config(config_path);paths=config['paths'];baseline=Path(paths['BASELINE_DIR']);profile=Path(paths['OUTPUT_DIR'])
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    watched=[baseline/(n+'.tif') for n in ('water','shadow_mask','layover_mask','terrain_exclusion')]
    before={str(p):sha256(p) for p in watched}
    metadata=json.loads((baseline/'metadata.json').read_text(encoding='utf-8'))
    settings=json.loads((baseline/'terrain_geometry.json').read_text(encoding='utf-8'))
    start=perf_counter();print('[01] 按原SAR网格生成方向修正掩膜',flush=True)
    generate_terrain_mask(baseline/'water.tif',paths['GUI_SLOPE_PATH'],paths['GUI_ASPECT_PATH'],metadata,output,
                          threshold=settings['slope_threshold_deg'],block_size=256)
    # 仅为通用对照函数提供相同的水体参考副本，文件内容不做任何修改。
    shutil.copy2(baseline/'water.tif',output/'water.tif')
    print('[02] 与既有剖面方法按同一30m网格对照',flush=True)
    paths['BASELINE_DIR']=output
    compare_scene(config,destination=output/'profile_comparison')
    summary={'common_domain_results':{},'fixed_direction_only':True,'elapsed_seconds':None}
    with rasterio.open(profile/'data_cache/reference.tif') as ref:
        valid=(ref.read(1)==1)&(warped_array(profile/'radar_coverage.tif',ref,'uint8',255)==1)
        for name in ('shadow_mask','layover_mask','terrain_exclusion'):
            old=warped_array(baseline/(name+'.tif'),ref,'uint8',255)
            fixed=warped_array(output/(name+'.tif'),ref,'uint8',255)
            new=warped_array(profile/(name+'.tif'),ref,'uint8',255)
            common=valid&(old<2)&(fixed<2)&(new<2)
            rows={}
            for label,a in [('legacy_local',old),('corrected_local',fixed)]:
                intersection=int((common&(a==1)&(new==1)).sum());union=int((common&((a==1)|(new==1))).sum())
                rows[label]=dict(local_pixels=int((common&(a==1)).sum()),profile_pixels=int((common&(new==1)).sum()),
                    intersection=intersection,union=union,positive_set_iou=intersection/union if union else None,
                    local_only=int((common&(a==1)&(new==0)).sum()),profile_only=int((common&(a==0)&(new==1)).sum()))
            summary['common_domain_results'][name]=dict(common_pixels=int(common.sum()),comparisons=rows)
    summary['elapsed_seconds']=perf_counter()-start
    summary['original_products_unchanged']=all(sha256(p)==before[str(p)] for p in watched)
    summary['original_sha256']=before
    write_json(output/'direction_comparison.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    return summary


if __name__=='__main__':main(config_argument())
