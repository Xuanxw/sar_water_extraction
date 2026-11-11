"""固定检查点的窗口上下文诊断。仅CPU局部推理，不训练、不修改模型权重或源数据。

对同一384×384区域，比较四个重叠1024窗口、固定ASPP池化向量、较大输入窗口。
固定池化向量只用于定位影响来源，不是正式推理策略；标签仅参与结果计量。
"""
from pathlib import Path
import gc
import json
import sys
import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.special import expit
import torch
import torch.nn.functional as F

PROJECT = Path(r"E:\Code_projects\sar_water_extraction")
sys.path.insert(0, str(PROJECT / "src"))
from supervised.models.checkpoint import load_model
from supervised.evaluation.metrics import metrics_from_counts
from supervised.evaluation.sliding_inference import blending_weights

CHECKPOINT = PROJECT / "outputs/pvt_deeplabv3plus_vv_sqrt_weight_bce_offline_normalized_bc_scenes_overlap/20261110_221430_589086/train_end.pth"
EXPECTED_EPOCH = 7
DATA_ROOT = Path(r"I:\水体数据\processed_scenes\normalized")
OUTPUT = PROJECT / "codex_analysis/window_context_epoch007_bn"
SCENE = "BC2-SM-ORG-VV-20221105T025841-003134"
# 从用户图件预先选取的上方矩形、下方台阶附近；不是全景或全验证集精度。
ROIS = ((1920, 4992, 384), (6912, 3840, 384))
CPU_THREADS = 2


def decode(model, features, output_size, fixed_context=None):
    """重用编码特征，仅在诊断分支替换ASPP全局池化向量。"""
    deep = features[-1]
    context = model.aspp.global_pool(deep) if fixed_context is None else fixed_context
    branches = [branch(deep) for branch in model.aspp.branches]
    context = F.interpolate(context, size=deep.shape[-2:], mode="bilinear", align_corners=False)
    high = model.aspp.project(torch.cat([*branches, context], dim=1))
    low = model.low_projection(features[0])
    high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
    logits = model.segmentation_head(model.fusion(torch.cat((high, low), dim=1)))
    return F.interpolate(logits, size=(output_size, output_size), mode="bilinear", align_corners=False)


def metrics(probability, label, valid):
    water, pred = label == 1, probability >= .5
    counts = [np.count_nonzero(valid & (water == a) & (pred == b)) for a,b in ((0,0),(0,1),(1,0),(1,1))]
    return metrics_from_counts(counts)


def save_roi_figure(path):
    """仅绘制小范围诊断，不把窗口批统计实验当成正式修复结果。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with np.load(path) as data:
        valid=data['valid']
        fig,axes=plt.subplots(1,5,figsize=(15,3.4))
        image=data['image'];lo,hi=np.percentile(image[valid],[2,98])
        axes[0].imshow(np.ma.array(image,mask=~valid),cmap='gray',vmin=lo,vmax=hi)
        axes[0].set_title('Input (display P2-P98)')
        axes[1].imshow(np.ma.array(data['label'],mask=~valid),cmap='gray_r',vmin=0,vmax=1)
        axes[1].set_title('GT water=1 black')
        keys=['hann_current','batch_bn_diagnostic_only','context_1536']
        titles=['Current Hann probability','Batch-BN DIAGNOSTIC ONLY','1536 context probability']
        for ax,key,title in zip(axes[2:],keys,titles):
            ax.imshow(np.ma.array(data[key],mask=~valid),cmap='Blues',vmin=0,vmax=1)
            ax.set_title(title)
        for ax in axes:ax.axis('off')
        fig.suptitle(path.stem+' | CPU FP32 | local diagnostic, not full validation')
        fig.tight_layout()
        try:fig.savefig(path.with_suffix('.png'),dpi=110)
        finally:plt.close(fig)


def main():
    torch.set_num_threads(CPU_THREADS)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if checkpoint['epoch'] != EXPECTED_EPOCH or not checkpoint.get('inputs_normalized'):
        raise ValueError("检查点已更新或不是离线归一化输入，请明确修改诊断配置")
    model = load_model(checkpoint).eval()
    report = dict(checkpoint=str(CHECKPOINT), epoch=checkpoint['epoch'],
                  validation_inference=checkpoint.get('validation_inference'),
                  scope='two_ROIs_CPU_FP32_not_full_validation', scene=SCENE, rois=[])
    del checkpoint
    gc.collect()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    weights = blending_weights(768, 'hann')
    with rasterio.Env(GDAL_CACHEMAX=64*1024**2), rasterio.open(DATA_ROOT/'images/val'/(SCENE+'.tif')) as src, \
            rasterio.open(DATA_ROOT/'labels/val'/(SCENE+'.tif')) as target, torch.inference_mode():
        def read(top,left,size):
            if min(top,left)<0 or top+size>src.height or left+size>src.width:
                raise ValueError('诊断窗口超出整景范围')
            data=src.read(1,window=Window(left,top,size,size),masked=True)
            good=~np.ma.getmaskarray(data)&np.isfinite(data.data)
            return torch.from_numpy(np.where(good,data.data,0).astype('float32'))[None,None]
        for top,left,size in ROIS:
            label=target.read(1,window=Window(left,top,size,size))
            raw=src.read(1,window=Window(left,top,size,size),masked=True)
            valid=~np.ma.getmaskarray(raw)&np.isfinite(raw.data)&np.isin(label,(0,1))
            features=[];contexts=[];predictions=[];tile_weights=[]
            origins=[(r,c) for r in (top-384,top) for c in (left-384,left)]
            for index,(r,c) in enumerate(origins):
                image=read(r-128,c-128,1024)
                feature=model.backbone(image)
                features.append(feature);contexts.append(model.aspp.global_pool(feature[-1]))
                ys,xs=slice(top-r+128,top-r+128+size),slice(left-c+128,left-c+128+size)
                out=decode(model,feature,1024)
                predictions.append(out[0,0,ys,xs].numpy().copy())
                tile_weights.append(weights[top-r:top-r+size,left-c:left-c+size])
                if index==0:
                    # 独立拆解的解码与正式forward必须一致，避免诊断本身引入错误。
                    torch.testing.assert_close(out,model(image),rtol=1e-5,atol=1e-5)
                print(f'ROI {top},{left} 原窗口 {index+1}/4',flush=True)
            fixed=torch.stack(contexts).mean(0)
            fixed_predictions=[]
            for feature,(r,c) in zip(features,origins):
                out=decode(model,feature,1024,fixed)
                fixed_predictions.append(out[0,0,top-r+128:top-r+128+size,left-c+128:left-c+128+size].numpy().copy())
            # 仅作BN敏感性诊断：不更新running统计、不打开dropout、不保存此模型。
            # 验证图像的批统计不得作为正式校准或泛化精度结论。
            bn_layers=[m for m in model.modules() if isinstance(m,torch.nn.BatchNorm2d)]
            bn_predictions=[]
            for bn in bn_layers:
                bn.train();bn.track_running_stats=False
            try:
                for feature,(r,c) in zip(features,origins):
                    out=decode(model,feature,1024)
                    bn_predictions.append(out[0,0,top-r+128:top-r+128+size,left-c+128:left-c+128+size].numpy().copy())
            finally:
                for bn in bn_layers:
                    bn.eval();bn.track_running_stats=True
            tile_weights=np.stack(tile_weights)
            fuse=lambda scores: (np.stack(scores)*tile_weights).sum(0)/tile_weights.sum(0)
            probabilities={'hann_current':expit(fuse(predictions)), 'fixed_pool_diagnostic':expit(fuse(fixed_predictions)), 'batch_bn_diagnostic_only':expit(fuse(bn_predictions))}
            for large_size in (1536,):
                margin=(large_size-size)//2
                out=model(read(top-margin,left-margin,large_size))
                probabilities[f'context_{large_size}']=torch.sigmoid(out[0,0,margin:margin+size,margin:margin+size]).numpy().copy()
            variants={}
            for name,scores in [('original',predictions),('fixed_pool',fixed_predictions),('batch_bn_diagnostic_only',bn_predictions)]:
                prob=expit(np.stack(scores));cls=prob>=.5
                variants[name]=dict(tile_mean_probability=[float(p[valid].mean()) for p in prob],
                    mean_probability_range=float((prob.max(0)-prob.min(0))[valid].mean()),
                    disagreement_ratio=float((cls.any(0)!=cls.all(0))[valid].mean()))
            result=dict(top=top,left=left,size=size,true_water_ratio=float((label[valid]==1).mean()),
                        windows=variants,metrics={k:metrics(v,label,valid) for k,v in probabilities.items()})
            report['rois'].append(result)
            np.savez_compressed(OUTPUT/f'roi_{top}_{left}.npz',image=raw.data,label=label,valid=valid,**probabilities)
            save_roi_figure(OUTPUT/f'roi_{top}_{left}.npz')
            (OUTPUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps(result,ensure_ascii=False),flush=True)
            del features,feature,contexts,out,image
            gc.collect()
    print('完成：',OUTPUT,flush=True)

if __name__=='__main__':
    main()
