"""最终成果留在outputs；前置栅格和源元数据放入data_cache，续跑记录随迁移更新。"""
import json
from pathlib import Path
NAMES=('raw.tif','water.tif','coverage.tif','history.tif','slope.tif','terrain_class.tif',
       'terrain_exclusion.tif','layover_mask.tif','shadow_mask.tif','terrain_geometry.json',
       'extraction.json','metadata.json','source.xml','input.json')


def arrange_intermediates(config,restore=False):
    """restore仅在执行阶段临时恢复工作输入；成功结束后重新归档，不重算影像。"""
    root=Path(config['paths']['OUTPUT_DIR']);cache=Path(config['paths']['CACHE_DIR'])
    count=0
    for scene in sorted(root.glob('*/*/scene.json')):
        folder=scene.parent;work=cache/'scenes'/folder.name/'products';work.mkdir(parents=True,exist_ok=True)
        for name in NAMES:
            src,dst=(work/name,folder/name) if restore else (folder/name,work/name)
            if src.exists():
                if not restore or not dst.exists():src.replace(dst);count+=1
        progress=work.parent/'progress.json'
        if progress.exists():
            state=json.loads(progress.read_text(encoding='utf-8'))
            for stage in state.get('stages',{}).values():
                for item in stage.get('files',[]):
                    if not restore and item['root']=='output' and item['path'] in NAMES:
                        item.update(root='cache',path='products/'+item['path'])
                    elif restore and item['root']=='cache' and item['path'].startswith('products/') and Path(item['path']).name in NAMES:
                        item.update(root='output',path=Path(item['path']).name)
            progress.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
        # JSON的来源路径跟随真实文件位置；统计值和面成果不改变。
        for name,key in (('water_polygon.json','source'),('quality.json','source_mask')):
            p=folder/name
            if p.exists():
                data=json.loads(p.read_text(encoding='utf-8'))
                if key in data and Path(data[key]).name in NAMES:
                    data[key]=str((folder if restore else work)/Path(data[key]).name)
                    p.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    print(('恢复工作输入' if restore else '中间成果移入data_cache')+f'：{count}个文件',flush=True)


def scene_product(folder,name):
    """从现有文件或质量记录的真实源路径定位前置产品，不绑定旧输出目录。"""
    folder=Path(folder);path=folder/name
    if path.exists():return path
    quality=json.loads((folder/'quality.json').read_text(encoding='utf-8'))
    return Path(quality['source_mask']).parent/name
