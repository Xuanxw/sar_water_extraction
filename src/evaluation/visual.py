"""在固定验证批次上观察 PVT-UANet/UAFNet 中间特征，不改变模型前向返回值。"""
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


class FeatureVisualizer:
    """临时挂接前向钩子，只保留前几个样本的二维响应，不存完整特征张量。"""

    def __init__(self, model, max_samples=2):
        self.model = model
        self.max_samples = max_samples
        self.maps = {}
        self.handles = []

    def capture(self, name, output):
        """沿通道取 mean(abs(feature))，立即转到 CPU，避免持有计算图。"""
        feature = output[:self.max_samples].detach().float()
        self.maps[name] = feature.abs().mean(dim=1).cpu().numpy()

    def __enter__(self):
        """仅在一次验证前向期间挂接钩子；原模块的输出不做修改。"""
        def capture_pyramid(prefix):
            def hook(module, inputs, outputs):
                for index, output in enumerate(outputs, 1):
                    self.capture(f"{prefix} {index}", output)
            return hook

        self.handles.append(self.model.backbone.register_forward_hook(capture_pyramid("PVT")))
        for index in range(2, 6):
            module = getattr(self.model, f"conv_{index}")
            self.handles.append(module.register_forward_hook(
                lambda module, inputs, output, index=index: self.capture(f"MBDC {index - 1}", output)))
        if hasattr(self.model, "neck"):
            self.handles.append(self.model.neck.register_forward_hook(capture_pyramid("FPN")))
        for index in (4, 3, 2):
            if hasattr(self.model, f"ufm_layer{index}"):
                module = getattr(self.model, f"ufm_layer{index}")
                # UAFM 返回 (融合特征, logits)，这里只取融合特征。
                self.handles.append(module.register_forward_hook(
                    lambda module, inputs, output, index=index: self.capture(f"UAFM {5 - index}", output[0])))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """即使前向报错也移除钩子；其余批次不承担特征采集开销。"""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def save(self, batch, predictions, folder, epoch, bands):
        """每幅样本保存一张面板图；特征各自拉伸，概率固定使用0～1。"""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        probabilities = []
        for output in (predictions[0], predictions[-1]):
            output = output[:self.max_samples].detach().float()
            probability = output[:, 0].sigmoid() if output.shape[1] == 1 else output.softmax(1)[:, 1]
            probabilities.append(probability.cpu().numpy())
        paths = []
        for sample in range(min(self.max_samples, len(batch["name"]))):
            fig = Figure(figsize=(14, 17), layout="constrained")
            FigureCanvasAgg(fig)  # 不启动GUI，也不改变已有 pyplot 的后端。
            axes = fig.subplots(5, 4)
            for ax in axes.flat:
                ax.set_axis_off()

            def panel(ax, values, title, stretch=False, valid=None, cmap="magma"):
                values = np.asarray(values, dtype=np.float32)
                mask = np.isfinite(values)
                if valid is not None:
                    mask &= valid
                low, high = 0., 1.
                if stretch and mask.any():
                    low, high = np.percentile(values[mask], (2, 98))
                    if high <= low:
                        high = low + 1.
                ax.imshow(np.ma.array(values, mask=~mask), cmap=cmap, vmin=low, vmax=high,
                          interpolation="nearest")
                ax.set_facecolor("#b0b0b0")
                ax.set_title(f"{title}\n{values.shape[0]}x{values.shape[1]} | [{low:.3g}, {high:.3g}]", fontsize=9)

            image = batch["image"][sample].cpu().numpy()
            valid = batch["valid_mask"][sample].cpu().numpy().astype(bool)
            label = batch["label"][sample].cpu().numpy()
            for channel in range(min(2, image.shape[0])):
                panel(axes[0, channel], image[channel], f"SAR band {bands[channel]} (stored values)",
                      stretch=True, valid=valid, cmap="gray")
            panel(axes[0, 2], label, "Ground truth", valid=label >= 0, cmap="Blues")
            panel(axes[0, 3], probabilities[1][sample], "Final water probability", valid=valid, cmap="viridis")
            for row, prefix in enumerate(("PVT", "MBDC", "FPN", "UAFM"), 1):
                for column in range(4 if prefix != "UAFM" else 3):
                    name = f"{prefix} {column + 1}"
                    if name in self.maps:
                        panel(axes[row, column], self.maps[name][sample], name, stretch=True)
            panel(axes[4, 3], probabilities[0][sample], "Coarse water probability", valid=valid, cmap="viridis")
            name = batch["name"][sample]
            fig.suptitle(f"Epoch {epoch:03d} | {name} | eval mode\n"
                         "Features: mean(abs(channels)), each panel P2-P98; brightness is NOT water probability",
                         fontsize=12)
            # 保留相对目录，避免不同子目录里的同名影像覆盖。
            path = folder / Path(name).with_suffix(".png")
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=110)
            fig.clear()
            paths.append(path)
        return paths
