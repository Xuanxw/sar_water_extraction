"""冻结共同前端的解码器机制实验；独立于正式 train.py，不代表完整重训练或跨域成绩。"""
import csv
import gc
import hashlib
import sys
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.water_dataset import WaterDataset
from models.checkpoint import load_model
from models.pvt import UAFM, NestedUNetDecoder
from models.losses import WaterSegmentationLoss
from evaluation.metrics import metrics_from_counts
from evaluation.model_eval import threshold_counts, precision_matches

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "uafm_front": ROOT / "outputs/pvt_uafnet_hard_sampling_residual_grouped/20261027_212638_633640/best_f1.pth",
    "nested_front": ROOT / "outputs/pvt_unetpp_hard_sampling_residual_grouped/20261028_105832_612811/best_f1.pth",
}
MODES = ("rank", "flat", "nested")
SEEDS = (42, 43, 44)
EPOCHS = 10
BATCH_SIZE = 8  # 机制小实验统一使用8；与正式训练batch=2的结果分开报告。
LEARNING_RATE = 1e-4
THRESHOLDS = [i / 100 for i in range(1, 100)]


class FlatRank(nn.Module):
    """取消预测引导，两个支路都保留原特征；卷积、通道和可训练参数量不变。"""
    def forward(self, logits):
        return torch.ones_like(logits), torch.ones_like(logits)


class RefinementProbe(nn.Module):
    """只训练细化部分，共有的四级特征和粗预测均来自冻结前端。"""
    def __init__(self, mode, channels=32):
        super().__init__()
        self.mode = mode
        if mode == "nested":
            self.nested = NestedUNetDecoder(channels)
            final_head = self.nested.head2
        else:
            self.layers = nn.ModuleList([UAFM(channels, 1) for _ in range(3)])
            if mode == "flat":
                for layer in self.layers:
                    layer.rank = FlatRank()
            final_head = self.layers[-1].seg_out
        nn.init.zeros_(final_head.weight)
        nn.init.zeros_(final_head.bias)

    def forward(self, features, coarse, size):
        if self.mode == "nested":
            p4, p3, delta = self.nested(features)
        else:
            f2, f3, f4, f5 = features
            guide = F.interpolate(coarse, size=f5.shape[-2:], mode="bilinear", align_corners=True)
            fusion4, p4 = self.layers[0](f4, f5, guide)
            fusion3, p3 = self.layers[1](f3, fusion4, p4)
            _, delta = self.layers[2](f2, fusion3, p3)
        # 与当前残差模型一致，最终输出是粗预测加修正；粗头冻结、无需反传其恒定损失。
        return tuple(F.interpolate(p, size=size, mode="bilinear", align_corners=True)
                     for p in (p4, p3, coarse + delta))


class FeatureCache(Dataset):
    """磁盘memmap按需读取，避免把约2.5GB特征装进内存。临时缓存用完即删除。"""
    def __init__(self, folder, names, train_count):
        self.names, self.train_count = names, train_count
        self.arrays = [np.load(Path(folder) / f"{key}.npy", mmap_mode="r")
                       for key in ("f2", "f3", "f4", "f5", "coarse", "label")]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        # copy仅复制这一张；避免可写性警告和Windows文件映射跨批持有。
        return tuple(torch.from_numpy(np.array(array[index], copy=True)) for array in self.arrays)

    def close(self):
        for array in self.arrays:
            array._mmap.close()
        self.arrays.clear()


def cache_frontend(checkpoint_path, folder, device):
    """只在eval模式提取train/val特征，所有细化方案共享完全相同的输入张量。"""
    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # 推理不需要优化器状态，尽早释放，避免占用用户正在工作的CPU内存。
    cp.pop("optimizer_state_dict", None)
    model = load_model(cp).to(device).eval().requires_grad_(False)
    names = cp["train_files"] + cp["val_files"]
    ntrain = len(cp["train_files"])
    mean = cp["mean"].to(device).view(1,-1,1,1)
    std = cp["std"].to(device).clamp_min(1e-6).view(1,-1,1,1)
    dataset = WaterDataset(cp["image_dir"], cp["label_dir"], cp["bands"])
    dataset.select_files(names)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    arrays = None
    cursor = 0
    with torch.inference_mode():
        for step, batch in enumerate(loader, 1):
            x = (batch["image"].to(device) - mean) / std
            x = x.masked_fill(~batch["valid_mask"].to(device).unsqueeze(1), 0)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                features = model.backbone(x)
                features = [getattr(model, f"conv_{i+2}")(f) for i,f in enumerate(features)]
                coarse = model.decoder(model.neck(features))
            values = [f.float().cpu().numpy().astype("float16") for f in features]
            values.append(coarse.float().cpu().numpy().astype("float16"))
            values.append(batch["label"].numpy().astype("int8"))
            if arrays is None:
                arrays = [np.lib.format.open_memmap(Path(folder) / f"{key}.npy", mode="w+",
                          dtype=value.dtype, shape=(len(names), *value.shape[1:]))
                          for key,value in zip(("f2","f3","f4","f5","coarse","label"),values)]
            for array,value in zip(arrays,values):
                array[cursor:cursor+len(x)] = value
            cursor += len(x)
            if step % 100 == 0:
                print(f"缓存冻结前端：{cursor}/{len(names)}", flush=True)
    for array in arrays:
        array.flush()
        array._mmap.close()
    del arrays, model, cp, values, features, coarse, x
    gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()
    return FeatureCache(folder, names, ntrain)


def evaluate_probe(model, loader, device, scan=False):
    """固定0.5阈值；最终轮附加共同Precision要求，绝不用val更新权重或BN。"""
    model.eval()
    counts = torch.zeros(4, dtype=torch.int64, device=device)
    sweep = torch.zeros(len(THRESHOLDS),4,dtype=torch.int64,device=device) if scan else None
    loss_sum = pixels = 0
    with torch.inference_mode():
        for batch in loader:
            features = [v.to(device, dtype=torch.float32) for v in batch[:4]]
            coarse = batch[4].to(device, dtype=torch.float32)
            label = batch[5].to(device, dtype=torch.long)
            valid = label != -1
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                score = model(features, coarse, label.shape[-2:])[-1][:,0].float()
            counts += torch.bincount(label[valid]*2 + (score[valid]>=0).long(), minlength=4)
            loss_sum += F.binary_cross_entropy_with_logits(score[valid],label[valid].float(),reduction="sum").item()
            pixels += int(valid.sum())
            if scan: sweep += threshold_counts(score,label,THRESHOLDS)
    metrics = dict(val_loss=loss_sum/pixels, **metrics_from_counts(counts.tolist()))
    rows = ([dict(threshold=t,**metrics_from_counts(c)) for t,c in zip(THRESHOLDS,sweep.tolist())]
            if scan else [])
    return metrics, rows


def run_probe(cache, source, mode, seed, folder, device, history_writer, final_writer, scan_writer):
    """相同seed使用相同初始卷积与固定随机遍历，不使用各模型自适应困难池或增强。"""
    torch.manual_seed(seed)
    model = RefinementProbe(mode).to(device)
    # 专用generator控制采样顺序，不受不同解码器的初始化或前向影响。
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(torch.utils.data.Subset(cache,range(cache.train_count)),
                              batch_size=BATCH_SIZE,shuffle=True,generator=generator,num_workers=0)
    val_loader = DataLoader(torch.utils.data.Subset(cache,range(cache.train_count,len(cache))),
                            batch_size=BATCH_SIZE,shuffle=False,num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    criterion = WaterSegmentationLoss("bce")
    # 零初始化最终头等于冻结coarse，记录它作为共同起点，避免只比较三个新头。
    initial,_=evaluate_probe(model,val_loader,device)
    history_writer.writerow(dict(source=source,mode=mode,seed=seed,epoch=0,
        train_loss=float("nan"),amp_skipped_steps=0,seconds=0,**initial))
    print(f"{source} {mode} seed={seed}: initial coarse F1={initial['f1']:.5f}",flush=True)
    for epoch in range(1,EPOCHS+1):
        start=perf_counter(); model.train()
        total_loss=total_pixels=skips=0
        for batch in train_loader:
            features=[v.to(device,dtype=torch.float32) for v in batch[:4]]
            coarse=batch[4].to(device,dtype=torch.float32)
            label=batch[5].to(device,dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=device.type=="cuda"):
                outputs=model(features,coarse,label.shape[-2:])
                loss=sum(criterion(output,label) for output in outputs)
            if not torch.isfinite(loss): raise FloatingPointError(f"{source}/{mode}/{seed}/{epoch}: nonfinite loss")
            scale=scaler.get_scale();scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
            skips += int(scaler.get_scale()<scale)
            count=int((label!=-1).sum());total_loss+=loss.item()*count;total_pixels+=count
        metrics,scan=evaluate_probe(model,val_loader,device,scan=epoch==EPOCHS)
        row=dict(source=source,mode=mode,seed=seed,epoch=epoch,train_loss=total_loss/total_pixels,
                 amp_skipped_steps=skips,seconds=perf_counter()-start,**metrics)
        history_writer.writerow(row)
        print(f"{source} {mode} seed={seed} epoch={epoch}/{EPOCHS}: F1={metrics['f1']:.5f}, "
              f"P={metrics['precision']:.5f}, R={metrics['recall']:.5f}, seconds={row['seconds']:.1f}",flush=True)
        scheduler.step()
    # 预先规定比较最终轮，不在val上挑选每个小实验的最佳轮次。
    matched=precision_matches(scan,(.8,.85,.9))
    final=dict(source=source,mode=mode,seed=seed,epoch=EPOCHS,parameters=sum(p.numel() for p in model.parameters()),**metrics)
    for item in matched:
        key=int(item['target_precision']*100)
        final[f"actual_precision_at_p{key}"]=item['precision']
        final[f"recall_at_p{key}"]=item['recall']
        final[f"threshold_at_p{key}"]=item['threshold']
    final_writer.writerow(final)
    for row in scan:scan_writer.writerow(dict(source=source,mode=mode,seed=seed,**row))
    torch.save(dict(source=source,mode=mode,seed=seed,epoch=EPOCHS,
                    model_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()}),
               folder/f"{source}_{mode}_seed{seed}.pth")
    del model,optimizer,scaler
    gc.collect()
    if device.type=="cuda":torch.cuda.empty_cache()


def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folder=ROOT/'codex_analysis/refinement_probe'/datetime.now().strftime('%Y%m%d_%H%M%S')
    folder.mkdir(parents=True)
    reference=None;provenance=[]
    for name,path in SOURCES.items():
        cp=torch.load(path,map_location='cpu',weights_only=True)
        identity={key:cp[key] for key in ('train_files','val_files','bands','split_manifest_sha256','image_dir','label_dir','ignore_index')}
        identity.update(mean=cp['mean'].tolist(),std=cp['std'].tolist())
        if reference is not None and identity!=reference:raise ValueError('两个前端的数据/归一化必须一致')
        reference=identity
        with path.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
        provenance.extend([f'{name}: {path}',f'sha256={digest}'])
        del cp
    (folder/'protocol.txt').write_text('\n'.join([
        'Conditional frozen-feature readout experiment. Not end-to-end training or external-domain evidence.',
        f'epochs={EPOCHS}; seeds={SEEDS}; batch={BATCH_SIZE}; lr={LEARNING_RATE}; AdamW wd=.01 cosine',
        'Freeze PVT/MBDC/FPN/coarse in eval mode. Cache original unaugmented views, float16 on disk.',
        'Same shuffled train sequence for paired seed; uniform sampling; no adaptive hard pool or augmentation.',
        'Train three refinement heads with equal BCE. Frozen coarse loss is constant and omitted.',
        'Primary result=final fixed epoch, not best-val checkpoint. All variants start at identical coarse final logits.',
        'rank vs flat: same conv initialization and parameter count; flat=both branch weights 1.',
        'nested: different topology/capacity, secondary comparison only.',
        'Predeclared practical signal: >=1 percentage-point F1 gain, same sign across 3 paired seeds and both frontends.',
        'A weak/negative result within this budget cannot exclude all end-to-end decoder improvements.',
        'Only existing grouped val; no geographic independence or cross-sensor generalization claim.',
        *provenance]),encoding='utf-8')
    metric_keys=list(metrics_from_counts((1,1,1,1)))
    with (folder/'history.csv').open('w',encoding='utf-8-sig',newline='',buffering=1) as h, \
         (folder/'final_metrics.csv').open('w',encoding='utf-8-sig',newline='',buffering=1) as f, \
         (folder/'threshold_metrics.csv').open('w',encoding='utf-8-sig',newline='',buffering=1) as s:
        hw=csv.DictWriter(h,fieldnames=['source','mode','seed','epoch','train_loss','amp_skipped_steps','seconds','val_loss',*metric_keys]);hw.writeheader()
        fw=csv.DictWriter(f,fieldnames=['source','mode','seed','epoch','parameters','val_loss',*metric_keys,*[item for p in (80,85,90) for item in (f'actual_precision_at_p{p}',f'recall_at_p{p}',f'threshold_at_p{p}')]]);fw.writeheader()
        sw=csv.DictWriter(s,fieldnames=['source','mode','seed','threshold',*metric_keys]);sw.writeheader()
        for source,path in SOURCES.items():
            # 缓存仅放在当前分析目录内；关闭映射后由TemporaryDirectory清理，不动用户数据。
            with TemporaryDirectory(prefix='feature_cache_',dir=folder) as temp:
                cache=cache_frontend(path,temp,device)
                try:
                    for seed in SEEDS:
                        for mode in MODES:
                            run_probe(cache,source,mode,seed,folder,device,hw,fw,sw)
                finally:
                    cache.close();del cache;gc.collect()
    print(f'PROBE_FINISHED {folder}',flush=True)
    return folder


if __name__=='__main__':
    main()
