"""为 train/val 全部 SAR TIFF 输出原尺寸快视图；不修改影像、标签或训练配置。

原始图：原幅度直接转8位灰度，不做逐景拉伸；超出0..255截断，小数量化。
标准化图：检查点 z=(x-mean)/std，固定显示范围。
每景只生成一张双面板PNG：左侧ORG，右侧标准化；两个面板均保留原始像素网格。
PNG 是8位显示产品，不是浮点训练输入；拼图不是可用于GIS分析的单景地理栅格。
"""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import shutil
import time
import os
import warnings
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.shutil import copy as raster_copy
from PIL import Image, ImageDraw, ImageFont
import torch

IMAGE_ROOT = Path(r"I:\水体数据\prepared_scenes\images")
CHECKPOINT = Path(r"E:\Code_projects\sar_water_extraction\outputs\pvt_deeplabv3plus_vv_balanced_bce_labels_corrected_bc_scenes\20261103_203233_244527\best_f1.pth")
SPLITS = ("val", "train")  # 先输出当前正在审查的验证景，再处理训练景。
BAND = 1
Z_DISPLAY_RANGE = (-1.0, 3.0)  # 只控制显示，不裁剪模型输入。所有景使用同一范围。
STRIPE_ROWS = 128             # 分条读取，避免把整景浮点数据放入内存。
FONT_PATH = Path(r"C:\Windows\Fonts\arial.ttf")
PNG_COMPRESSION = 3


def display_bytes(values, valid, low, high):
    """有效数值映射到1..255，0仅用于显示无效区；有效原始0值仍然保留。"""
    scaled=np.clip((values-low)/max(high-low,1e-6),0,1)
    return np.where(valid,1+254*np.nan_to_num(scaled,nan=0,posinf=1,neginf=0),0).astype('uint8')


def export_scene(path, mean, std, checkpoint_hash):
    """分条写入左右两个原尺寸面板，只输出一张PNG；JSON最后落盘用于续跑。"""
    output=path.parent/'quicklooks'
    output.mkdir(exist_ok=True)
    figure=output/(path.stem+'.comparison.png')
    metadata=output/(path.stem+'.comparison.json')
    signature=dict(version=3,source=str(path),source_size=path.stat().st_size,
                   source_mtime_ns=path.stat().st_mtime_ns,checkpoint_sha256=checkpoint_hash,
                   band=BAND,mean=mean,std=std,z_display_range=list(Z_DISPLAY_RANGE),
                   org_display="direct_uint8_no_stretch",org_display_range=[0,255])
    if metadata.exists() and figure.exists():
        saved=json.loads(metadata.read_text(encoding='utf-8'))
        if saved.get('configuration')==signature:
            return dict(name=path.name,status='reused',**saved['statistics'])
    with rasterio.open(path) as src:
        low,high=0.,255.  # ORG直接显示幅度值，不估计/应用每景百分位或增益。
        # 字体随影像宽度放大；标题和间隔仅增加画布，不缩放或重采样SAR面板。
        font_size=max(12,src.width//100)
        font=ImageFont.truetype(str(FONT_PATH),font_size) if FONT_PATH.exists() else ImageFont.load_default()
        margin=max(8,font_size//2)
        line=font_size+margin
        title_height=6*line
        gap=max(16,src.width//100)
        canvas_width=2*src.width+gap
        canvas_height=src.height+title_height
        pixels=src.width*src.height
        if shutil.disk_usage(output).free < canvas_width*canvas_height*2+2*1024**3:
            raise OSError('磁盘余量不足以安全输出下一景')
        # 临时TIFF只是分块画布，不携带虚假的坐标系。转换PNG后删除。
        temp=output/(path.stem+'.comparison.work.tif')
        profile=dict(driver='GTiff',width=canvas_width,height=canvas_height,count=1,dtype='uint8',
                     tiled=True,blockxsize=256,blockysize=256,compress='deflate',BIGTIFF='IF_SAFER')
        valid_count=zero_count=z_low_count=z_high_count=0
        raw_low_count=raw_high_count=0
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',rasterio.errors.NotGeoreferencedWarning)
                with rasterio.open(temp,'w',**profile) as dst:
                    header=Image.new('L',(canvas_width,title_height),255)
                    draw=ImageDraw.Draw(header)
                    draw.text((margin,margin),path.stem,fill=0,font=font)
                    draw.text((margin,line),f'Each panel: {src.width} x {src.height} native pixels | 8-bit display only',fill=0,font=font)
                    draw.text((margin,2*line),'No filtering / radiometric correction. Values below/above display limits are clipped only for display.',fill=0,font=font)
                    draw.text((margin,3*line),'ORG / original amplitude (no stretch)',fill=0,font=font)
                    draw.text((margin,4*line),'Direct 8-bit gray: [0,255]; decimals quantized, values >255 clipped',fill=0,font=font)
                    right=src.width+gap+margin
                    draw.text((right,3*line),'Normalized / training mean and std',fill=0,font=font)
                    draw.text((right,4*line),f'z=(DN-{mean:.6f})/{std:.6f}',fill=0,font=font)
                    draw.text((right,5*line),f'Fixed display z range: {Z_DISPLAY_RANGE}; valid zero retained',fill=0,font=font)
                    dst.write(np.asarray(header),1,window=Window(0,0,canvas_width,title_height))
                    for top in range(0,src.height,STRIPE_ROWS):
                        h=min(STRIPE_ROWS,src.height-top)
                        a=src.read(BAND,window=Window(0,top,src.width,h),masked=True,out_dtype='float32')
                        valid=~np.ma.getmaskarray(a)&np.isfinite(a.data)
                        z=(a.data-mean)/std
                        # 画布同一行：原始显示、白色分隔、标准化显示。
                        stripe=np.full((h,canvas_width),255,np.uint8)
                        # ORG不乘增益、不减最小值，原幅度直接量化到灰度；有效0保持黑色。
                        stripe[:,:src.width]=np.where(valid,np.clip(
                            np.nan_to_num(a.data,nan=0,posinf=255,neginf=0),0,255),0).astype('uint8')
                        stripe[:,src.width+gap:]=display_bytes(z,valid,*Z_DISPLAY_RANGE)
                        dst.write(stripe,1,window=Window(0,title_height+top,canvas_width,h))
                        valid_count+=int(valid.sum())
                        zero_count+=int((valid&(a.data==0)).sum())
                        z_low_count+=int((valid&(z<Z_DISPLAY_RANGE[0])).sum())
                        z_high_count+=int((valid&(z>Z_DISPLAY_RANGE[1])).sum())
                        raw_low_count+=int((valid&(a.data<low)).sum())
                        raw_high_count+=int((valid&(a.data>high)).sum())
                raster_copy(str(temp),str(figure),driver='PNG',ZLEVEL=PNG_COMPRESSION)
            statistics=dict(width=src.width,height=src.height,figure_width=canvas_width,
                            figure_height=canvas_height,valid_pixels=valid_count,
                            invalid_pixels=pixels-valid_count,valid_zero_pixels=zero_count,
                            raw_display_min=float(low),raw_display_max=high,
                            raw_below_display=raw_low_count,raw_above_display=raw_high_count,
                            z_below_display=z_low_count,z_above_display=z_high_count)
            metadata.write_text(json.dumps(dict(configuration=signature,statistics=statistics,
                panels=dict(org=[0,title_height,src.width,src.height],
                            normalized=[src.width+gap,title_height,src.width,src.height]),
                source_crs=str(src.crs),source_transform=list(src.transform),
                note='One comparison figure, not a geospatial raster. Both panels preserve native pixels. '
                     'ORG directly quantized to uint8 [0,255], no adaptive stretching. '
                     'Normalized panel has a fixed display mapping; model float inputs are not clipped. '
                     'ORG panel uses the stored SAR TIFF values, not a newly imported source product.'),
                ensure_ascii=False,indent=2),encoding='utf-8')
        finally:
            if temp.exists():
                temp.unlink()  # 仅清理本函数的临时拼图；原SAR和旧快视图均保留。
    return dict(name=path.name,status='written',**statistics)


def main():
    """全量输出train/val实际存在的TIFF（包括清单中未采用的影像），每景记录进度。"""
    torch.set_num_threads(1)
    checkpoint=torch.load(CHECKPOINT,map_location='cpu',weights_only=False,mmap=True)
    mean=float(torch.as_tensor(checkpoint['mean']).flatten()[0])
    std=float(torch.as_tensor(checkpoint['std']).flatten()[0])
    # 本脚本只复现当前未滤波、单VV检查点；不把显示变换冒充训练变换。
    if tuple(checkpoint.get('bands',())) != (BAND,) or torch.as_tensor(checkpoint['mean']).numel()!=1:
        raise ValueError('当前脚本只支持本次单VV检查点')
    if checkpoint.get('speckle') not in (None,'none'):
        raise ValueError('检查点包含滤波，请先适配相同滤波流程，不能仅生成标准化图')
    with CHECKPOINT.open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    files=[p for split in SPLITS for p in sorted((IMAGE_ROOT/split).iterdir())
           if p.is_file() and p.suffix.lower() in ('.tif','.tiff')]
    print(f'双面板快视图：{len(files)}景；mean={mean}；std={std}；输出为各split下quicklooks',flush=True)
    started=time.time(); done=errors=0
    failures=[]
    for split in SPLITS:
        output=IMAGE_ROOT/split/'quicklooks'
        output.mkdir(exist_ok=True)
        (output/'快视图说明.txt').write_text(
            '每景仅输出一张 *.comparison.png，左侧ORG，右侧normalized。另有参数记录 *.comparison.json。\n'
            '两个面板均保持原始SAR的宽高和像素排列；拼图另增加标题和分隔带，不做降采样。\n'
            'ORG使用当前SAR TIFF原幅度直接转8位灰度，不自动拉伸：小数量化，低于0/超过255截断。normalized使用训练检查点均值和标准差、固定z范围[-1,3]显示。\n'
            'PNG仅为8位显示图，不是浮点训练输入，不可将整张拼图当作单景地理栅格分析。\n'
            '不做滤波、辐射校正或dB变换；有效0值保留。ORG中的有效0和无效区均显示黑色，不能仅凭黑色判断无效。\n'
            'ORG直接灰度与normalized固定范围显示不同，不能用两面板亮度差认定辐射异常；浮点幅度请读取原始TIFF。\n'
            '旧版raw/normalized/overview文件不会再生成；已存在的旧文件不自动删除。\n',encoding='utf-8')
    with rasterio.Env(GDAL_CACHEMAX=64*1024**2,GDAL_NUM_THREADS='1'):
        for index,path in enumerate(files,1):
            try:
                row=export_scene(path,mean,std,digest)
                done+=1
                print(f'{index}/{len(files)} [{path.parent.name}] {path.stem}：{row["status"]} '
                      f'{row["width"]}x{row["height"]}',flush=True)
            except Exception as error:
                errors+=1
                failures.append(dict(image=str(path),error=str(error)))
                print(f'{index}/{len(files)} 失败：{path}：{error}',flush=True)
            state=dict(pid=os.getpid(),complete=index==len(files) and errors==0,processed=index,successful=done,
                       errors=errors,expected=len(files),last_image=str(path),
                       elapsed_seconds=round(time.time()-started,1),updated=datetime.now().isoformat())
            state['failures']=failures
            for split in SPLITS:
                (IMAGE_ROOT/split/'quicklooks'/'comparison_status.json').write_text(
                    json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'快视图结束：成功{done}/{len(files)}，失败{errors}',flush=True)


if __name__=='__main__':
    main()
