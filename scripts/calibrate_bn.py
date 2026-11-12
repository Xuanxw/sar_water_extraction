"""只用训练影像逐层校准 BN，比较前后验证指标，不重新训练权重。"""

import csv
import json
from time import perf_counter
from datetime import datetime
from pathlib import Path
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

# 支持按文件路径直接运行，也支持包导入。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets.dfc_sample.s1_dataset import WaterDataset, sample_name
from supervised.models.checkpoint import load_model
from supervised.evaluation.sliding_inference import iter_scene_predictions
from supervised.evaluation.metrics import metrics_from_counts
from supervised.load.sar_scene_dataset import SARSceneDataset, check_preprocessed_input
import numpy as np
import hashlib

ROOT = Path("E:/Code_projects/sar_water_extraction")
# 固定要诊断的源检查点；训练仍在运行时train_end可能被下一轮覆盖，请先停止或改用固定副本。
RUN_DIR = ROOT / "outputs/pvt_deeplabv3plus_vv_sqrt_weight_bce_offline_normalized_bc_scenes_overlap/20261110_221430_589086"
CHECKPOINT_NAME = "train_end.pth"
# None新建对照；指定目录则复用校准模型，跳过CSV中已经完成的景/阶段。
RESUME_DIR = RUN_DIR / "bn_calibration/20261111_164753_515753"
EXPECTED_EPOCH = 7  # 防止运行中的train_end更新后，误把其他轮当成第7轮；None不限定。
CALIBRATION_SAMPLES = 128  # 固定种子抽取train窗口，不能按val表现挑选。
CACHE_DIR = ROOT / "data_cache/bc_scenes"
CPU_THREADS = 2
# ()评价全部val；指定影像编号时仅做局部诊断，不代表完整验证集成绩。
EVALUATION_SCENES = ()
# 仅调整评价顺序，仍覆盖全部val，先检查用户报告的场景。
PRIORITY_SCENE = "BC2-SM-ORG-VV-20221105T025841-003134"



class ChannelStatistics:
    """单层 BN 的输入统计器；作为前向钩子累计逐通道一阶、二阶矩。"""

    def __init__(self, channels, device):
        self.channel_sum = torch.zeros(channels, dtype=torch.float64, device=device)
        self.channel_square_sum = torch.zeros_like(self.channel_sum)
        self.pixel_count = 0

    def __call__(self, module, inputs):
        """PyTorch 在 BN 前向之前调用，inputs[0] 是 [B,C,H,W] 特征。"""
        feature = inputs[0].double()  # 只在统计时用双精度，不改变网络前向精度。
        self.channel_sum.add_(feature.sum(dim=(0, 2, 3)))
        self.channel_square_sum.add_(feature.square().sum(dim=(0, 2, 3)))
        self.pixel_count += feature.shape[0] * feature.shape[2] * feature.shape[3]

    def apply_to(self, layer):
        """将累计的 E[x]、E[x²] 换算为均值和方差，写入当前 BN 缓冲区。"""
        mean = self.channel_sum / self.pixel_count
        var = (self.channel_square_sum / self.pixel_count - mean.square()).clamp_min(0)
        layer.running_mean.copy_(mean)
        layer.running_var.copy_(var)


class BNCalibrator:
    """BN 校准器：加载模型与原划分，只用训练影像逐层恢复运行统计量。"""

    def __init__(self):
        """加载源模型、固定训练校准子集和完整验证集。"""
        self.run_dir = RUN_DIR
        self.run_dir = Path(self.run_dir)
        self.checkpoint_path = self.run_dir / CHECKPOINT_NAME
        self.checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        if EXPECTED_EPOCH is not None and self.checkpoint['epoch'] != EXPECTED_EPOCH:
            raise ValueError(f"源检查点已变为第{self.checkpoint['epoch']}轮；预期第{EXPECTED_EPOCH}轮，请核对配置")
        self.validation_inference = self.checkpoint.get('validation_inference', {})
        self.output_dir = (Path(RESUME_DIR) if RESUME_DIR is not None else
                           self.run_dir / "bn_calibration" / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        if RESUME_DIR is not None:
            if not (self.output_dir/'calibrated.pth').is_file():
                raise FileNotFoundError(f'续跑目录缺少calibrated.pth：{self.output_dir}')
        else:
            self.output_dir.mkdir(parents=True, exist_ok=False)
        # 校准只能恢复统计量不匹配，不能修复已经变成 NaN/Inf 的模型参数。
        if not all(torch.isfinite(value).all().item() for value in self.checkpoint["model_state_dict"].values()):
            raise ValueError("模型包含 NaN/Inf，不能通过 BN 校准恢复，请选择有限权重的 BN 模型。")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(self.checkpoint).to(self.device)  # 根据模型名称重建并严格加载。
        self.model.eval()

        # 完全使用检查点保存的划分；标签不参与BN统计量估计。
        if self.checkpoint.get('data_layout') == 'full_scene':
            root = Path(self.checkpoint['image_dir']).parent
            check_preprocessed_input(root/'images',self.checkpoint)
            if hashlib.sha256((root/'split_manifest.csv').read_bytes()).hexdigest() != self.checkpoint['split_manifest_sha256']:
                raise ValueError('BN校准清单与检查点不同')
            revision = root/'revision_settings.json'
            if self.checkpoint.get('label_revision_sha256') and hashlib.sha256(revision.read_bytes()).hexdigest() != self.checkpoint['label_revision_sha256']:
                raise ValueError('BN校准标签版本与检查点不同')
            train_data = val_data = SARSceneDataset(root, CACHE_DIR,
                bands=self.checkpoint['bands'], window_size=self.checkpoint['window_size'],
                halo=self.checkpoint['window_halo'], speckle=self.checkpoint.get('speckle'))
            file_indices = {name:i for i,name in enumerate(train_data.names)}
            # 固定随机训练窗口；不按val表现挑选校准数据，不使用val统计。
            candidates = self.checkpoint['train_files']
            rng = np.random.default_rng(self.checkpoint.get('seed',42))
            selected = rng.choice(len(candidates),min(CALIBRATION_SAMPLES,len(candidates)),replace=False)
            self.calibration_files = [candidates[i] for i in selected]
            train_ids = [file_indices[name] for name in self.calibration_files]
            val_ids = [file_indices[name] for name in self.checkpoint['val_files']]
            if not set(train_ids)<=set(train_data.train_ids) or not set(val_ids)<=set(train_data.val_ids):
                raise ValueError('校准/验证窗口越过原划分')
        else:
            train_data = WaterDataset(self.checkpoint['image_dir'], bands=self.checkpoint['bands'],speckle=self.checkpoint.get('speckle'))
            val_data = WaterDataset(self.checkpoint['image_dir'],self.checkpoint['label_dir'],self.checkpoint['bands'],speckle=self.checkpoint.get('speckle'))
            file_indices = {sample_name(path,train_data.image_dir):i for i,path in enumerate(train_data.image_paths)}
            self.calibration_files = self.checkpoint['train_files'][:CALIBRATION_SAMPLES]
            train_ids = [file_indices[name] for name in self.calibration_files]
            val_ids = [file_indices[name] for name in self.checkpoint['val_files']]
        batch_size = self.checkpoint["batch_size"]
        self.train_loader = DataLoader(Subset(train_data, train_ids), batch_size=batch_size, shuffle=False)
        self.val_loader = DataLoader(Subset(val_data, val_ids), batch_size=batch_size, shuffle=False)
        self.mean = self.checkpoint["mean"].to(self.device).view(1, -1, 1, 1)
        self.std = self.checkpoint["std"].to(self.device).view(1, -1, 1, 1)
        self.ignore_index = self.checkpoint["ignore_index"]
        self.dataset = val_data
        print(f"模型：{self.checkpoint_path}；设备：{self.device}；校准/验证：{len(train_ids)}/{len(val_ids)}", flush=True)

    def evaluate_blended(self, scene_ids=None):
        """沿用源检查点的融合协议；每个有效像素只计一次，并输出逐景指标。"""
        self.model.eval()
        config = self.validation_inference
        names = set(self.checkpoint['val_scene_files'])
        scenes = [row for row in self.dataset.scenes if row['split']=='val' and row['name'] in names]
        selected = EVALUATION_SCENES if scene_ids is None else scene_ids
        if selected:
            scenes = [row for row in scenes if row['scene_id'] in selected]
            if {r['scene_id'] for r in scenes} != set(selected):
                raise ValueError('指定的评价景不在源检查点val划分中')
        total=np.zeros(4,np.int64)
        loss_sum=0.0
        self.scene_metrics=[]
        for number,row in enumerate(scenes,1):
            counts=np.zeros(4,np.int64)
            scene_loss=0.0
            blocks=0
            print(f"BN对照融合验证：{number}/{len(scenes)}景 {row['scene_id']}",flush=True)
            for block in iter_scene_predictions(self.model,
                    self.dataset.root/'images'/row['split']/row['name'],
                    self.dataset.root/'labels'/row['split']/row['label_name'],
                    bands=self.checkpoint['bands'],window_size=self.checkpoint['window_size'],
                    halo=self.checkpoint['window_halo'],stride=config.get('stride'),blend=config['blend'],
                    mean=self.mean.cpu().flatten().numpy(),std=self.std.cpu().flatten().numpy(),
                    inputs_normalized=self.checkpoint.get('inputs_normalized',False),device=self.device,
                    use_amp=self.device.type=='cuda' and self.checkpoint.get('amp_enabled',False),
                    batch_size=self.checkpoint['batch_size'],num_workers=config.get('num_workers',0),
                    speckle=self.checkpoint.get('speckle')):
                valid=block['valid_mask'] & np.isin(block['label'],(0,1))
                y=block['label'][valid].astype(np.uint8)
                z=block['scores'][valid]
                counts+=np.bincount(2*y+(z>=0).astype(np.uint8),minlength=4)
                # 单logit BCE与双类别CE均可由二类logit差计算；不加训练类别权重。
                scene_loss+=float(np.sum(np.logaddexp(0,z)-y*z,dtype=np.float64))
                blocks+=1
                if blocks%100==0:
                    print(f"  {getattr(self,'phase','val')}：已输出{blocks}个融合块",flush=True)
            total+=counts
            loss_sum+=scene_loss
            self.scene_metrics.append(dict(scene_id=row['scene_id'],loss=scene_loss/int(counts.sum()) if counts.sum() else float('nan'),**metrics_from_counts(counts)))
        if total.sum()==0:
            raise ValueError('没有可评价的val像素')
        return dict(loss=float(loss_sum)/int(total.sum()),**metrics_from_counts(total))

    def evaluate(self):
        """用标准eval和检查点AMP配置，计算完整验证集指标。"""
        if self.validation_inference:
            return self.evaluate_blended()
        self.model.eval()
        confusion = torch.zeros(4, dtype=torch.int64, device=self.device)
        loss_sum, pixel_count = 0.0, 0
        with torch.no_grad():
            for batch in self.val_loader:
                image = batch["image"].to(self.device)
                if not self.checkpoint.get("inputs_normalized",False):
                    image = (image - self.mean) / self.std
                image = image.masked_fill(~batch["valid_mask"].to(self.device).unsqueeze(1), 0)
                label = batch["label"].to(self.device)
                valid = label != self.ignore_index
                count = int(valid.sum().item())
                if count == 0:
                    continue
                # 与训练验证保持相同AMP设置，避免把数值精度差异混入BN对照。
                with torch.autocast(device_type=self.device.type, enabled=self.device.type=='cuda' and self.checkpoint.get('amp_enabled',False)):
                    outputs = self.model(image)
                logits = outputs[-1] if isinstance(outputs, (tuple, list)) else outputs
                if logits.shape[1] == 1:
                    loss = nn.functional.binary_cross_entropy_with_logits(logits[:,0].float()[valid],label[valid].float())
                    prediction = (logits[:,0]>=0).long()
                else:
                    loss = nn.functional.cross_entropy(logits.float(),label,ignore_index=self.ignore_index)
                    prediction = logits.argmax(1)
                if not torch.isfinite(logits).all().item() or not torch.isfinite(loss).item():
                    raise FloatingPointError(f"验证出现非有限输出或 loss；文件：{batch['name']}")
                confusion += torch.bincount(label[valid] * 2 + prediction[valid], minlength=4)
                loss_sum += loss.item() * count
                pixel_count += count
        tn, fp, fn, tp = confusion.tolist()
        return dict(
            loss=loss_sum / pixel_count,
            iou=tp / (tp + fp + fn) if tp + fp + fn else float("nan"),
            f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan"),
            precision=tp / (tp + fp) if tp + fp else 0.0,
            recall=tp / (tp + fn) if tp + fn else float("nan"),
            tn=tn, fp=fp, fn=fn, tp=tp, valid_pixels=pixel_count,
            true_water_ratio=(tp + fn) / pixel_count,
            pred_water_ratio=(tp + fp) / pixel_count,
        )

    def calibrate_layers(self):
        """依次重估各层 BN 输入统计量，不把所有 BN 同时切到 train。"""
        # 当前模型的 BN 注册顺序满足上游先于下游；PVT 的 LayerNorm 不参与校准。
        # 上游先校准，所有层始终保持 eval，不更新任何可学习参数。
        for name, layer in self.model.named_modules():
            if not isinstance(layer, nn.BatchNorm2d):
                continue
            statistics = ChannelStatistics(layer.num_features, self.device)
            handle = layer.register_forward_pre_hook(statistics)
            try:
                with torch.no_grad():
                    for step,batch in enumerate(self.train_loader,1):
                        image = batch["image"].to(self.device)
                        if not self.checkpoint.get("inputs_normalized",False):
                            image = (image - self.mean) / self.std
                        image = image.masked_fill(~batch["valid_mask"].to(self.device).unsqueeze(1), 0)
                        self.model(image)
                        if step%16==0:
                            print(f"校准 {name}：{step}/{len(self.train_loader)}批",flush=True)
                    statistics.apply_to(layer)
            finally:
                handle.remove()
            print(f"已校准：{name}", flush=True)

    def save_result(self, before, after):
        """另存校准模型和前后指标，保留原始检查点。"""
        # 校准产物单独保存，不覆盖原始模型，不按验证成绩反复挑选校准样本。
        output_dir = self.output_dir
        with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=["phase", *before.keys()])
            writer.writeheader()
            writer.writerow(dict(phase="before", **before))
            writer.writerow(dict(phase="after", **after))
        self.save_calibrated_model(before,after)

    def save_calibrated_model(self, before=None, after=None):
        """校准完成立即落盘；完整评价完成后才填入验证成绩。"""
        output_dir=self.output_dir
        checkpoint=dict(self.checkpoint)
        checkpoint["model_state_dict"] = {name: value.cpu() for name, value in self.model.state_dict().items()}
        # 保留原归一化类型，PVT 同时包含 LayerNorm 和解码头的 BatchNorm。
        checkpoint.setdefault("normalization", "BatchNorm")
        checkpoint["source_val_loss"] = checkpoint.get("val_loss")
        checkpoint["source_phase_metrics"] = checkpoint.get("phase_metrics", {})
        checkpoint["source_best_val_loss"] = checkpoint.pop("best_val_loss", None)
        checkpoint["val_loss"] = after["loss"] if after else None
        checkpoint["phase_metrics"] = {"val": after} if after else {}
        checkpoint["bn_calibration"] = {
            "method": "sequential_train_feature_moments", "source_checkpoint": str(self.checkpoint_path),
            "train_files": self.calibration_files, "batch_size": checkpoint["batch_size"], "precision": "float32",
            "before": before, "after": after, "comparison_complete": after is not None,
            "validation_inference": self.validation_inference,
            "evaluation_scenes": list(EVALUATION_SCENES),
            "evaluation_scope": 'selected_val_scenes' if EVALUATION_SCENES else 'all_val',
        }
        # BN已改变，这不是原实验的可续跑状态；只用于独立推理对照。
        checkpoint['resume_phase'] = 'bn_calibrated_inference_only'
        for key in ('optimizer_state_dict','scheduler_state_dict','scaler_state_dict','rng_state',
                    'best_val_f1','best_val_balanced_accuracy'):
            checkpoint.pop(key,None)
        torch.save(checkpoint, output_dir / "calibrated.pth.partial")
        (output_dir / "calibrated.pth.partial").replace(output_dir / "calibrated.pth")
        print(f"校准模型和前后指标：{output_dir}", flush=True)

    def run(self):
        """先校准并保存，再逐景成对评价；每景立即写CSV和进度。"""
        print(f"输出目录：{self.output_dir}",flush=True)
        started=perf_counter()
        original=self.checkpoint['model_state_dict']
        if RESUME_DIR is not None:
            saved=torch.load(self.output_dir/'calibrated.pth',map_location='cpu',weights_only=True)
            meta=saved['bn_calibration']
            for key in ('epoch','model_name','model_kwargs','bands','split_manifest_sha256',
                        'label_revision_sha256','input_preprocessing','validation_inference'):
                if saved.get(key)!=self.checkpoint.get(key):
                    raise ValueError(f'续跑源模型或数据协议不一致：{key}')
            if (meta['train_files']!=self.calibration_files or
                    meta['evaluation_scenes']!=list(EVALUATION_SCENES) or
                    Path(meta['source_checkpoint']).resolve()!=self.checkpoint_path.resolve()):
                raise ValueError('续跑校准样本、评价范围或源检查点不一致')
            calibrated=saved['model_state_dict']
            self.model.load_state_dict(calibrated)
            del saved
            print('续跑：已加载校准模型，不重新校准BN。',flush=True)
        else:
            self.calibrate_layers()
            calibrated={k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()}
        # 校准只允许改变BN运行统计；禁止意外改变学习参数。
        for name,value in self.model.named_parameters():
            if not torch.equal(value.detach().cpu(),original[name]):
                raise RuntimeError(f'校准意外改变了学习参数：{name}')
        if RESUME_DIR is None:
            self.save_calibrated_model()
        if not self.validation_inference:
            self.model.load_state_dict(original);before=self.evaluate()
            self.model.load_state_dict(calibrated);after=self.evaluate()
        else:
            names=set(self.checkpoint['val_scene_files'])
            scenes=[r['scene_id'] for r in self.dataset.scenes if r['split']=='val' and r['name'] in names]
            if EVALUATION_SCENES:
                if not set(EVALUATION_SCENES)<=set(scenes):
                    raise ValueError('指定场景不在val中')
                scenes=[n for n in scenes if n in EVALUATION_SCENES]
            scenes.sort(key=lambda n:n!=PRIORITY_SCENE)
            # 按景和阶段恢复：若某景before完成、after中断，只补after，不重复计数。
            completed={p:{} for p in ('before','after')}
            if RESUME_DIR is not None:
                for phase in completed:
                    path=self.output_dir/f'{phase}_per_scene.csv'
                    if path.exists():
                        with path.open(encoding='utf-8-sig',newline='') as f:
                            for row in csv.DictReader(f):
                                scene=row.pop('scene_id')
                                if scene not in scenes or scene in completed[phase]:
                                    raise ValueError(f'已存结果存在重复或非本次val场景：{scene}')
                                completed[phase][scene]={k:int(v) if k in ('tn','fp','fn','tp','valid_pixels')
                                                        else float(v) for k,v in row.items()}
                paired=set(completed['before']) & set(completed['after'])
                print(f'续跑：保留{len(paired)}/{len(scenes)}景完整对照，继续未完成阶段。',flush=True)
            progress_path=self.output_dir/'progress.json'
            previous_elapsed=(json.loads(progress_path.read_text(encoding='utf-8')).get('elapsed_seconds',0)
                              if RESUME_DIR is not None and progress_path.exists() else 0)
            counts={p:np.zeros(4,np.int64) for p in ('before','after')}
            losses=dict(before=0.0,after=0.0)
            for i,scene in enumerate(scenes,1):
                pair={}
                for phase,state in (('before',original),('after',calibrated)):
                    self.phase=phase
                    if scene in completed[phase]:
                        result=completed[phase][scene]
                    else:
                        print(f"对照 {i}/{len(scenes)}景 {phase}：{scene}",flush=True)
                        self.model.load_state_dict(state)
                        result=self.evaluate_blended((scene,))
                        row=dict(scene_id=scene,**result)
                        path=self.output_dir/f'{phase}_per_scene.csv'
                        write_header=not path.exists() or path.stat().st_size==0
                        with path.open('a',encoding='utf-8-sig',newline='') as f:
                            writer=csv.DictWriter(f,fieldnames=list(row))
                            if write_header:writer.writeheader()
                            writer.writerow(row)
                    pair[phase]=result
                    counts[phase]+=np.array([result[k] for k in ('tn','fp','fn','tp')],np.int64)
                    losses[phase]+=result['loss']*result['valid_pixels']
                if pair['before']['valid_pixels']!=pair['after']['valid_pixels']:
                    raise RuntimeError('前后有效像素数不一致')
                print(f"完成 {i}/{len(scenes)}景：F1 {pair['before']['f1']:.4f} → {pair['after']['f1']:.4f}；"
                      f"P {pair['before']['precision']:.4f} → {pair['after']['precision']:.4f}；"
                      f"R {pair['before']['recall']:.4f} → {pair['after']['recall']:.4f}",flush=True)
                progress=dict(completed_scenes=i,total_scenes=len(scenes),last_scene=scene,
                    complete=i==len(scenes),elapsed_seconds=previous_elapsed+perf_counter()-started)
                (self.output_dir/'progress.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2),encoding='utf-8')
            before,after=[dict(loss=losses[p]/int(counts[p].sum()),**metrics_from_counts(counts[p])) for p in ('before','after')]
        self.model.load_state_dict(calibrated)
        self.save_result(before,after)
        print(f"校准前：{before}\n校准后：{after}",flush=True)
        for loader in (self.train_loader,self.val_loader):
            data=loader.dataset.dataset
            if hasattr(data,'close'):data.close()


if __name__ == "__main__":
    torch.set_num_threads(CPU_THREADS)
    calibrator = BNCalibrator()
    calibrator.run()
