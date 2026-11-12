"""对已有阈值栅格补做GUI式面清理；原water.tif/shp保持不变。"""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from configuration import load_yaml
from thres_segmentation.org_io import export_water_polygon,write_json,read_aoi,prepare_auxiliary

CONFIG_PATH=Path(r'E:\Code_projects\sar_water_extraction\configs\北京密云区\threshold_segmentation_20261112.yaml')

def main(config_path=CONFIG_PATH):
    config=load_yaml(config_path);root=Path(config['paths']['OUTPUT_DIR'])
    from thres_segmentation.product_layout import arrange_intermediates
    arrange_intermediates(config,restore=True)
    # 补成果只复用本地辅助缓存，不下载数据、不运行SNAP和阈值计算。
    config['org']['auxiliary']['download_missing']=False
    aoi,crs=read_aoi(config['paths']['AOI_PATH'],config['org']['output_crs'])
    auxiliary=prepare_auxiliary(config,aoi,crs)
    kwargs=dict(options=config.get('polygon_output',{}),postprocess=config['org']['postprocess'],
                cover_path=auxiliary.get('worldcover'),slope_path=auxiliary.get('slope'),clip_geometry=aoi,clip_crs=crs)
    scenes=sorted(root.glob('*/*/scene.json'));completed={}
    print(f'开始补水体面：{len(scenes)}景；输出在原逐景目录；参数：{kwargs["options"]}',flush=True)
    for index,path in enumerate(scenes,1):
        record=json.loads(path.read_text(encoding='utf-8'))
        if record.get('status')!='complete':continue
        print(f'[{index}/{len(scenes)}] {record["scene_id"]}',flush=True)
        destination=export_water_polygon(path.parent/'water.tif',**kwargs)
        record['water_polygon_shp']=str(destination.resolve());record['water_shp']=str(destination.resolve());write_json(path,record)
        completed[record['scene_id']]=str(destination.resolve())
    summary=root/'results.json'
    if summary.exists():
        rows=json.loads(summary.read_text(encoding='utf-8'))
        for row in rows:
            if row['scene_id'] in completed:row.update(water_polygon_shp=completed[row['scene_id']],water_shp=completed[row['scene_id']])
        write_json(summary,rows)
    if config['org'].get('archive_intermediates',False):arrange_intermediates(config)
    print(f'全部完成：{len(completed)}景。原water.tif未修改；每景water_polygon.json记录清理效果。',flush=True)

if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else CONFIG_PATH)
