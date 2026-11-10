"""直接复用训练内val预测的质量图；不再次推理，不保存网络中间层。"""
import csv
import gc
import json
from pathlib import Path

import numpy as np
import rasterio
import torch
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from supervised.evaluation.metrics import metrics_from_counts


def save_prediction_figure(*args, **kwargs):
    """每景结束及时回收Figure/Canvas循环引用，避免图件内存跨景累积。"""
    try:
        _render_prediction_figure(*args, **kwargs)
    finally:
        gc.collect()


def _render_prediction_figure(image, label, probability, prediction, counts, path, title):
    """图像是概览，标题指标来自原生有效像素；未知/未验证位置不冒充背景。"""
    valid = label >= 0
    error = np.zeros((*label.shape, 3), dtype=np.uint8)
    error[valid] = (45, 45, 45)
    error[valid & (label == 1) & (prediction == 1)] = (40, 140, 255)
    error[valid & (label == 0) & (prediction == 1)] = (255, 60, 60)
    error[valid & (label == 1) & (prediction == 0)] = (255, 220, 40)
    fig = Figure(figsize=(21, 5)); FigureCanvasAgg(fig)
    axes = fig.subplots(1, 5)
    finite = image[np.isfinite(image)]
    lo, hi = np.percentile(finite, [2, 98]) if finite.size else (0., 1.)
    axes[0].imshow(image, interpolation="nearest", interpolation_stage="data", cmap="gray", vmin=lo, vmax=max(hi, lo+1e-6))
    axes[1].imshow(np.ma.masked_where(~valid, label), interpolation="nearest", interpolation_stage="data", cmap="gray_r", vmin=0, vmax=1)
    axes[2].imshow(probability, interpolation="nearest", interpolation_stage="data", cmap="Blues", vmin=0, vmax=1)
    axes[3].imshow(np.ma.masked_where(prediction < 0, prediction), interpolation="nearest", interpolation_stage="data", cmap="gray_r", vmin=0, vmax=1)
    axes[4].imshow(error)
    for ax, caption in zip(axes, ("Model input (display P2-P98)", "GT: water=1 black", "Water probability", "Raw prediction (threshold=0.5)", "Blue TP / Red FP / Yellow FN")):
        ax.set_title(caption); ax.axis("off")
    m = metrics_from_counts(counts)
    fig.suptitle(f"{title} | native evaluated pixels={m['valid_pixels']:,} | F1={m['f1']:.3f} "
                 f"IoU={m['iou']:.3f} P={m['precision']:.3f} R={m['recall']:.3f} | overview; not a full-scene product")
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    fig.clear()


class ValidationQualityWriter:
    """按val顺序逐景拼概览，只保留当前景的小数组；全分辨率计数独立累计。"""
    def __init__(self, dataset, indices, folder, epoch, max_side=1200, max_scenes=None):
        self.dataset, self.folder, self.epoch = dataset, Path(folder), epoch
        self.max_side, self.max_scenes = max(1, int(max_side)), max_scenes
        self.lookup = {dataset.names[int(i)]: int(i) for i in indices} if hasattr(dataset, "windows") else None
        self.current, self.rows, self.counts = None, [], np.zeros(4, dtype=np.int64)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.status = self.folder / "status.json"
        self.status.write_text(json.dumps(dict(complete=False, epoch=epoch)), encoding="utf-8")

    def add(self, name, image, label, probability, prediction, valid_mask, scene_window=None, scene_shape=None, counts=None):
        """取本批已计算的预测；整景模式仅拼入中心区，halo不会重复计数。"""
        if scene_window is not None:
            top, left, h, w = scene_window
            key, halo = name, 0  # 共享推理器已裁边并融合，不能再次裁halo。
        elif self.lookup is not None:
            sid, top, left, h, w, *_ = map(int, self.dataset.windows[self.lookup[name]])
            row = self.dataset.scenes[sid]; key = row['scene_id']; halo = self.dataset.halo
        else:
            key, top, left, halo = name, 0, 0, 0
            h, w = label.shape
        if key != self.current:
            self.flush()
            self.current = key
            self.selected = self.max_scenes is None or len(self.rows) < self.max_scenes
            if not self.selected:
                return
            if scene_window is not None:
                height, width = scene_shape
            elif self.lookup is not None:
                # 仅读栅格头获取整景大小，不额外读取SAR像素。
                with rasterio.open(self.dataset.root/'images'/row['split']/row['name']) as src:
                    height, width = src.shape
            else:
                height, width = h, w
            stride = max(1, int(np.ceil(max(height, width) / self.max_side)))
            self.ys, self.xs = np.arange(0, height, stride), np.arange(0, width, stride)
            shape = (len(self.ys), len(self.xs))
            self.image = np.full(shape, np.nan, dtype='float32')
            self.probability = np.full(shape, np.nan, dtype='float32')
            self.label = np.full(shape, -1, dtype='int8')
            self.prediction = np.full(shape, -1, dtype='int8')
            self.counts = np.zeros(4, dtype=np.int64)
        if not self.selected:
            return
        iy = np.flatnonzero((self.ys >= top) & (self.ys < top+h))
        ix = np.flatnonzero((self.xs >= left) & (self.xs < left+w))
        dst = np.ix_(iy, ix)
        tensors = (image, label, probability, prediction, valid_mask)
        dtypes = (torch.float32, torch.int8, torch.float32, torch.int8, torch.bool)
        if counts is not None:
            # 原分辨率计数来自验证器；只复制概览实际需要的像素，避免整块重复搬运/计量。
            self.counts += np.asarray(counts, dtype=np.int64)
            if not len(iy) or not len(ix):
                return
            previews = []
            for tensor, dtype in zip(tensors, dtypes):
                rows = torch.as_tensor(self.ys[iy]-top+halo, device=tensor.device)
                cols = torch.as_tensor(self.xs[ix]-left+halo, device=tensor.device)
                previews.append(tensor.detach()[rows[:, None], cols[None, :]].to(dtype=dtype).cpu().numpy())
            image, label, probability, prediction, valid_mask = previews
            self.image[dst] = np.where(valid_mask, image, np.nan)
            self.label[dst] = label
            self.probability[dst] = np.where(valid_mask, probability, np.nan)
            self.prediction[dst] = np.where(valid_mask, prediction, -1)
            return
        # GPU只复制已有本批结果，不保留计算图；不改动训练随机数或模型状态。
        # 在复制到CPU前压缩类别数组；不修改训练中原有tensor。
        image, label, probability, prediction, valid_mask = [
            x.detach().to(dtype=dtype).cpu().numpy() for x, dtype in zip(tensors, dtypes)
        ]
        core = label[halo:halo+h, halo:halo+w]
        pred = prediction[halo:halo+h, halo:halo+w]
        # 按64行计数，避免整窗口布尔索引产生多个int64临时数组。
        # 仍统计全部有效中心像素；与概览降采样无关。
        for start in range(0, h, 64):
            truth = core[start:start+64]
            result = pred[start:start+64]
            for actual in (0, 1):
                for predicted in (0, 1):
                    self.counts[2*actual+predicted] += np.count_nonzero(
                        (truth == actual) & (result == predicted))
        dst = np.ix_(iy, ix); src = np.ix_(self.ys[iy]-top+halo, self.xs[ix]-left+halo)
        image_valid = valid_mask[src]
        self.image[dst] = np.where(image_valid, image[src], np.nan)
        self.label[dst] = label[src]
        self.probability[dst] = np.where(image_valid, probability[src], np.nan)
        self.prediction[dst] = np.where(image_valid, prediction[src], -1)

    def flush(self):
        """一景结束即写图并释放其数组；中断时status保留complete=false。"""
        if self.current is None or not self.selected:
            return
        name = str(self.current).replace("/", "__").replace(chr(92), "__")
        path = self.folder/'figures'/(name+'.png')
        save_prediction_figure(self.image,self.label,self.probability,self.prediction,self.counts,
                               path,f"Epoch {self.epoch:03d} val | {self.current}")
        self.rows.append(dict(scene_id=self.current, **metrics_from_counts(self.counts)))
        self.current = None
        # 保存结束即释放上一景概览，不等到下一景数组分配后再替换。
        self.image = self.label = self.probability = self.prediction = None

    def finish(self):
        """只有完整val循环结束才发布本轮图件完成标记。"""
        self.flush()
        if self.rows:
            with (self.folder/'per_scene.csv').open('w',encoding='utf-8-sig',newline='') as stream:
                writer=csv.DictWriter(stream, fieldnames=self.rows[0]);writer.writeheader();writer.writerows(self.rows)
        self.status.write_text(json.dumps(dict(complete=True,epoch=self.epoch,scenes=len(self.rows),
            scope='validation_evaluated_cores; preview_only',max_scenes=self.max_scenes)),encoding='utf-8')
        print(f"训练验证质量图：{self.folder}（{len(self.rows)}景）",flush=True)


def save_training_curves(*args, **kwargs):
    """训练曲线也在保存后立即回收绘图对象。"""
    try:
        _render_training_curves(*args, **kwargs)
    finally:
        gc.collect()


def _render_training_curves(metrics_path, output_path):
    """loss/F1/IoU/P/R来自训练日志，不重新评价模型。"""
    with Path(metrics_path).open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    fig=Figure(figsize=(14,4)); FigureCanvasAgg(fig); axes=fig.subplots(1,3)
    for phase in ('train','val'):
        selected=[r for r in rows if r['phase']==phase]
        epochs=[int(r['epoch']) for r in selected]
        axes[0].plot(epochs,[float(r['loss']) for r in selected],label=phase)
        for key in ('f1','iou'):
            axes[1].plot(epochs,[float(r[key]) for r in selected],label=f'{phase} {key}')
        for key in ('precision','recall'):
            axes[2].plot(epochs,[float(r[key]) for r in selected],label=f'{phase} {key}')
    for ax,title in zip(axes,('Loss (see weighting config)','F1 / IoU','Precision / Recall')):
        ax.set_title(title);ax.set_xlabel('Epoch');ax.legend();ax.grid(alpha=.25)
    fig.tight_layout(); output_path=Path(output_path);output_path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output_path,dpi=110);fig.clear()
