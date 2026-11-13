"""独立入口：inspect检查、nominal名义几何、visibility整景候选掩膜；不启动水体生产。"""
from pathlib import Path
from datetime import datetime
import sys
import traceback
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from configuration import config_argument
from sar_geometry.metadata import read_bc_metadata, make_model
from sar_geometry.terrain_io import run_nominal, write_json



def load_config(config_path):
    """入口统一读取路径；验证脚本复用同一解析逻辑。"""
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding='utf-8-sig'))
    # 所有显式相对路径以YAML所在目录为基准；不读取其他模块的配置。
    def resolve(value):
        """普通路径以YAML为基准；显式GDAL网络路径原样传递，不将URL变成本地路径。"""
        if value is None:return None
        if isinstance(value,list):return [resolve(v) for v in value]
        if str(value).startswith(('/vsicurl/','https://')):return value
        p=Path(value).expanduser()
        return p if p.is_absolute() else (config_path.parent/p).resolve()
    config['paths']={k:resolve(v) for k,v in config['paths'].items()}
    return config


def main(config_path):
    config=load_config(config_path)
    output = config['paths']['OUTPUT_DIR'];output.mkdir(parents=True,exist_ok=True)
    def log(message):
        line=f'{datetime.now().astimezone().isoformat(timespec="seconds")} {message}'
        print(line,flush=True)
        with (output/'processing.log').open('a',encoding='utf-8') as f:
            f.write(line+'\n')
    try:
        log(f'[01] 读取BC注释；模式={config["mode"]}')
        metadata=read_bc_metadata(config['paths']['XML_PATH'],config.get('xml_member'))
        info=dict(**metadata['description'],radar_grid=metadata['grid'],
                  orbit_time_range_s=[float(metadata['orbit'][0,0]),float(metadata['orbit'][-1,0])],
                  doppler_time_range_s=([float(metadata['doppler'][0,0]),float(metadata['doppler'][-1,0])]
                                        if len(metadata['doppler']) else None),
                  radar_grid_confirmed=config['model']['radar_grid_confirmed'],
                  selected_doppler_mode=config['model']['doppler_mode'],height_datum=config['model']['height_datum'])
        if config['mode']=='inspect':
            write_json(output/'metadata_inspection.json',info)
            log(f'[02] 注释检查完成：{info["orbit_records"]}个轨道状态；雷达网格{info["radar_grid"]["columns"]}×{info["radar_grid"]["rows"]}')
            log('[02] 仅列出字段；请确认网格、Doppler约定和DEM高程基准后再运行nominal。')
            return info
        if config['mode']=='visibility':
            from sar_geometry.scene import run_visibility
            from sar_geometry.scene_validation import ResourceMonitor
            import rasterio
            # 限制GDAL缓存；监测的是父进程及短生命周期C++子进程的RSS总和。
            with ResourceMonitor() as usage, rasterio.Env(GDAL_CACHEMAX=64*1024*1024):
                report=run_visibility(config,metadata,log)
            if not report.get('memory_monitor'):
                report['memory_monitor']=usage.result()
                write_json(output/'geometry_quality.json',report)
            return report
        if config['mode']!='nominal':
            raise ValueError('mode应为inspect/nominal/visibility')
        if config['paths']['DEM_PATH'] is None:
            raise ValueError('nominal需要明确DEM_PATH')
        model=make_model(metadata,config['model'])
        log('[02] 开始名义几何；不计算整景阴影或替换水体产品')
        result=run_nominal(config,metadata,model,log)
        log('[04] 名义几何完成');return result
    except Exception:
        log('[ERROR] '+traceback.format_exc());raise


if __name__=='__main__':
    main(config_argument())
