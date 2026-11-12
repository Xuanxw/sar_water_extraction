"""无真值质量检查：分块统计、分层随机核验点、疑似风险区域；不计算精度。"""
import json
from contextlib import ExitStack
from pathlib import Path

import fiona
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt
from thres_segmentation.water_analysis import pixel_areas, area_sum


def load_review_options():
    """质量检查参数独立配置，不读取训练配置、不启动影像处理。"""
    import yaml
    path=Path(__file__).resolve().parents[2]/'configs/threshold_segmentation.yaml'
    with path.open(encoding='utf-8') as stream:
        return yaml.safe_load(stream)['quality_review']


def export_quality_review(mask_path, output_dir, *, scene_id=None, points_per_stratum=50,
                          boundary_radius_px=5, seed=42, block_size=1024,
                          history_path=None, slope_path=None, raw_mask_path=None,
                          slope_threshold_deg=15., metadata=None, log=print):
    """输入最终分类栅格：0非水体、1水体、其他值/NoData未知。

    辅助栅格需提前对齐：history/raw为0/1，slope为度；不猜类别编码或重采样。
    每层按像元随机优先级抽样，所有候选像元等概率，无放回；与是否疑似错误无关。
    风险面分块输出，块边界不代表真实斑块边界，不以面数量统计水体个数。
    """
    if points_per_stratum < 0 or boundary_radius_px < 1 or block_size < 1:
        raise ValueError('采样点数不能为负，边界半径与分块大小必须为正整数')
    mask_path, output_dir = Path(mask_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_id = scene_id or mask_path.stem
    vector_path = output_dir/'review.gpkg'
    json_path = output_dir/'quality.json'
    # 同一输出目录不混用旧核验结果；由调用入口使用新的实验目录。
    if vector_path.exists() or json_path.exists():
        raise FileExistsError(f'核验成果已存在，请另选输出目录：{output_dir}')
    rng = np.random.default_rng(seed)
    names=('water_interior','shore_buffer','nonwater_other')
    populations=dict.fromkeys(names,0)
    samples={name:[] for name in names}
    risk_counts=dict.fromkeys(('water_near_unknown','history_disagreement','steep_water','postprocess_changed'),0)
    risk_areas=dict.fromkeys(risk_counts,0.)
    valid_pixels=water_pixels=0
    valid_area=water_area=0.
    with ExitStack() as stack:
        src=stack.enter_context(rasterio.open(mask_path))
        if src.crs is None:
            raise ValueError('地理核验矢量需要有效CRS')
        auxiliary={}
        for key,path in (('history',history_path),('slope',slope_path),('raw',raw_mask_path)):
            if path is None:
                continue
            aux=stack.enter_context(rasterio.open(path))
            if (aux.shape,aux.crs,aux.transform)!=(src.shape,src.crs,src.transform):
                raise ValueError(f'{key}必须与最终分类栅格同网格')
            auxiliary[key]=aux
        schema={'geometry':'Polygon','properties':{'scene_id':'str','reason':'str'}}
        risks=stack.enter_context(fiona.open(vector_path,'w',driver='GPKG',layer='suspected_regions',
                                            schema=schema,crs_wkt=src.crs.to_wkt()))
        halo=int(boundary_radius_px)
        for top in range(0,src.height,block_size):
            for left in range(0,src.width,block_size):
                height,width=min(block_size,src.height-top),min(block_size,src.width-left)
                x0,y0=max(0,left-halo),max(0,top-halo)
                x1,y1=min(src.width,left+width+halo),min(src.height,top+height+halo)
                win=Window(x0,y0,x1-x0,y1-y0)
                data=src.read(1,window=win,masked=True)
                valid=~np.ma.getmaskarray(data)&np.isin(data.data,(0,1))
                water=valid&(data.data==1);dry=valid&(data.data==0)
                # 只在已知水陆之间定义岸线，不把影像外缘和NoData边界当成岸线。
                shore=np.zeros(valid.shape,bool)
                if water.any() and dry.any():
                    shore=valid&(((distance_transform_edt(~water)<=halo)&dry)|
                                 ((distance_transform_edt(~dry)<=halo)&water))
                # 影像外部也属于未知；padding保证外缘可被风险检查识别。
                unknown_distance=distance_transform_edt(np.pad(valid,1,constant_values=False))[1:-1,1:-1]
                crop=np.s_[top-y0:top-y0+height,left-x0:left-x0+width]
                core_valid=valid[crop];core_water=water[crop]
                core_transform=src.window_transform(Window(left,top,width,height))
                areas=pixel_areas(src.crs,core_transform,(height,width))
                valid_pixels+=int(core_valid.sum());water_pixels+=int(core_water.sum())
                valid_area+=area_sum(core_valid,areas);water_area+=area_sum(core_water,areas)
                layers=(water&~shore,shore,dry&~shore)
                for name,layer in zip(names,layers):
                    ys,xs=np.nonzero(layer[crop]);n=len(ys)
                    populations[name]+=n
                    if n==0 or points_per_stratum==0:
                        continue
                    priorities=rng.random(n)
                    k=min(points_per_stratum,n)
                    selected=np.argpartition(priorities,k-1)[:k]
                    # 只保留当前块及历史候选中优先级最小的k个，内存与场景大小无关。
                    samples[name].extend((float(priorities[i]),int(top+ys[i]),int(left+xs[i]),
                                          int(core_water[ys[i],xs[i]])) for i in selected)
                    samples[name]=sorted(samples[name])[:points_per_stratum]
                flags={'water_near_unknown':(water&(unknown_distance<=halo))[crop]}
                for key,aux in auxiliary.items():
                    values=aux.read(1,window=Window(left,top,width,height),masked=True)
                    known=~np.ma.getmaskarray(values)&np.isfinite(values.data)&core_valid
                    if key=='slope':
                        flags['steep_water']=known&core_water&(values.data>slope_threshold_deg)
                    else:
                        known &= np.isin(values.data,(0,1))
                        reason='history_disagreement' if key=='history' else 'postprocess_changed'
                        flags[reason]=known&(core_water!=(values.data==1))
                for reason,flag in flags.items():
                    risk_counts[reason]+=int(flag.sum());risk_areas[reason]+=area_sum(flag,areas)
                    # 面仅供定位核查；历史差异可能是真实变化，坡度不等于雷达阴影。
                    risks.writerecords({'geometry':geom,'properties':{'scene_id':scene_id,'reason':reason}}
                        for geom,value in shapes(flag.astype('uint8'),mask=flag,transform=core_transform))
            log(f'质量核查：{scene_id} 行 {min(top+block_size,src.height)}/{src.height}')
        crs_wkt=src.crs.to_wkt();transform=src.transform
        extent_pixels=src.width*src.height
    point_schema={'geometry':'Point','properties':{'scene_id':'str','stratum':'str','row':'int',
        'col':'int','pred':'int','ref_class':'int','confidence':'str','note':'str',
        'population':'int64','sample_n':'int','incl_prob':'float','weight':'float'}}
    with fiona.open(vector_path,'w',driver='GPKG',layer='verification_points',
                    schema=point_schema,crs_wkt=crs_wkt) as dst:
        for name,records in samples.items():
            n=len(records);total=populations[name]
            for _,row,col,pred in records:
                x,y=transform*(col+.5,row+.5)
                dst.write({'geometry':{'type':'Point','coordinates':(x,y)},'properties':{
                    'scene_id':scene_id,'stratum':name,'row':row,'col':col,'pred':pred,
                    'ref_class':None,'confidence':'','note':'','population':total,'sample_n':n,
                    'incl_prob':n/total,'weight':total/n}})
    checks={key:{'status':'computed' if key=='water_near_unknown' or
        {'history_disagreement':'history','steep_water':'slope','postprocess_changed':'raw'}[key] in auxiliary
        else 'not_available', 'pixels':risk_counts[key], 'area_m2':risk_areas[key]} for key in risk_counts}
    for item in checks.values():
        if item['status']=='not_available':item.update(pixels=None,area_m2=None)
    report={'scene_id':scene_id,'assessment':'quality_only_without_ground_truth',
        'source_mask':str(mask_path.resolve()),'metadata':metadata or {},
        'coverage':{'scope':'mask_raster_extent_not_administrative_area','valid_pixels':valid_pixels,
                    'unknown_pixels':extent_pixels-valid_pixels,'valid_area_m2':valid_area},
        'water':{'pixels':water_pixels,'area_m2':water_area,
                 'fraction_of_valid':water_pixels/valid_pixels if valid_pixels else None},
        'sampling':{'seed':seed,'block_size':block_size,'boundary_radius_px':halo,'strata':{
            name:{'population':populations[name],'sample_n':len(samples[name])} for name in names},
            'reference_codes':'0=nonwater,1=water,-1=uncertain,null=not_reviewed',
            'weight_unit':'pixels; geographic grids need area weighting for area inference'},
        'checks':checks,'not_evaluated':['threshold_sensitivity','overlap_consistency','temporal_change'],
        'accuracy':{'status':'not_evaluated_no_reference','f1':None,'iou':None},
        'artifacts':{'vectors':'review.gpkg','layers':['verification_points','suspected_regions']},
        'notes':['疑似区域是核查候选，不是已确认错误；不同风险面可能重叠，不能相加统计。',
                 '风险面可能在处理块边界分开，面数量不代表水体斑块数量。',
                 'NoData为未知而非非水体；核验点覆盖三层，不仅在疑似区域抽样。']}
    json_path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return report
