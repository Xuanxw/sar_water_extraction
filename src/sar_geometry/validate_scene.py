"""手动运行一次整景计算及新旧掩膜对照；保留阶段日志，不启动后台批处理。"""
from pathlib import Path
import sys
from time import perf_counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from configuration import config_argument
from sar_geometry.main import main,load_config
from sar_geometry.scene_validation import compare_scene,compare_sampling
from sar_geometry.terrain_io import sha256,write_json



def validate(config_path):
    """以原有文件哈希确认未被替换，输出几何差异及包含图件的总耗时。"""
    config=load_config(config_path);baseline=Path(config['paths']['BASELINE_DIR']);out=Path(config['paths']['OUTPUT_DIR'])
    originals={p:sha256(p) for p in [baseline/(n+'.tif') for n in ('water','shadow_mask','layover_mask','terrain_exclusion')]}
    started=perf_counter();quality=main(config_path)
    print('[06] 同网格对照和图件输出',flush=True);comparison=compare_scene(config)
    elapsed=perf_counter()-started
    summary=dict(elapsed_including_preparation_comparison_seconds=elapsed,time_budget_seconds=config['execution']['time_budget_seconds'],
                 cache_reused_this_run=quality.get('cache_reused_this_run',False),
                 end_to_end_budget_met=elapsed<=config['execution']['time_budget_seconds'],
                 original_products_unchanged=all(sha256(p)==digest for p,digest in originals.items()),
                 quality=quality,comparison= comparison['counts'])
    write_json(out/'validation.json',summary)
    print(f'[07] 对照完成：{elapsed:.2f}s；原有成果未变={summary["original_products_unchanged"]}',flush=True)
    return summary


if __name__=='__main__':
    selected=config_argument(optional_args=1)
    validate(selected)
    # 可选第二个YAML仅用于已有粗采样成果对照；不重新运行它，不改变其文件。
    if len(sys.argv)>2:
        coarse=load_config(sys.argv[2]);fine=load_config(selected)
        print(compare_sampling(coarse['paths']['OUTPUT_DIR'],fine['paths']['OUTPUT_DIR']),flush=True)
