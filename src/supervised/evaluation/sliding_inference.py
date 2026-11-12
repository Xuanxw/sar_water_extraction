"""带上下文的重叠滑窗推理：先融合 logits，再交给调用者统一阈值化。

按行滚动累积，两个融合缓冲均只有 core × scene_width，不载入整幅影像。
标签只在最终输出时读取，不参与窗口选择、权重生成或模型预测。
"""
from contextlib import ExitStack

import numpy as np
import rasterio
from rasterio.windows import Window
import torch
from torch.utils.data import DataLoader

from datasets.speckle import SpeckleFilter, read_filtered_window
from supervised.load.scene_windows import SceneWindows


def blending_weights(core, blend):
    """Hann/Gaussian 强调核心区中央并保留正边缘权重；constant 为等权融合。"""
    if blend == 'constant':
        return np.ones((core, core), np.float32)
    if blend == 'hann':
        axis = np.hanning(core).astype(np.float32)
        return np.maximum(axis[:, None] * axis[None, :], 1e-3).astype(np.float32)
    if blend != 'gaussian':
        raise ValueError('blend 只支持 gaussian、hann 或 constant')
    coordinate = (np.arange(core, dtype=np.float32) - (core - 1) / 2) / max(core / 4, 1)
    axis = np.exp(-0.5 * coordinate ** 2)
    return np.maximum(axis[:, None] * axis[None, :], 1e-3).astype(np.float32)


def iter_scene_predictions(model, image_path, label_path=None, *, bands=(1,),
                           window_size=1024, halo=128, stride=None, blend='gaussian',
                           mean=None, std=None, inputs_normalized=False, device='cpu',
                           use_amp=False, batch_size=1, num_workers=0, speckle=None,
                           boundary_context=True):
    """逐块返回最终融合结果，每个原分辨率像素仅输出一次。

    scores 为单通道 logit（两类时为 logit1-logit0）；image 已位于模型输入域。
    label_context/valid_context 提供额外 halo，core_slices 定位当前输出块，
    用于在拼接线两侧连续计算真值边界。无标签时 label 全为 -1。
    boundary_context=False 时上下文仅含当前核心块，可复用窗口中的 SAR/有效掩膜；
    此时不适用于需要跨块邻域的边界指标。缓存高度仍只有 core。
    """
    # 保留标签网格核验，但重叠输入窗口不读取/分配未被模型使用的标签。
    ds = SceneWindows(image_path, label_path, bands, window_size, halo, speckle, stride, read_labels=False)
    device = torch.device(device)
    means = torch.as_tensor(mean if mean is not None else np.zeros(len(bands)),
                            device=device, dtype=torch.float32).reshape(1, -1, 1, 1)
    stds = torch.as_tensor(std if std is not None else np.ones(len(bands)),
                           device=device, dtype=torch.float32).clamp_min(1e-6).reshape(1, -1, 1, 1)
    means_np = means.cpu().numpy()[0]
    stds_np = stds.cpu().numpy()[0]
    weights = blending_weights(ds.core, blend)
    # 只保留尚可能被下一行窗口更新的条带；不创建整景临时文件。
    sums = np.zeros((ds.core, ds.width), np.float32)
    totals = np.zeros_like(sums)
    # 仅原始读取及已准备的整景 SNAP 缓存与读取窗口无关；未知/在线滤波安全回退。
    reuse_input = not boundary_context and type(ds.filter) is SpeckleFilter and (
        ds.filter.config is None or ds.filter.snap is not None)
    cached_image = np.zeros((len(bands), ds.core, ds.width), np.float32) if reuse_input else None
    cached_valid = np.zeros((ds.core, ds.width), bool) if reuse_input else None
    start_row = 0
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=use_amp and device.type == 'cuda',
                        generator=torch.Generator().manual_seed(0))
    was_training = model.training
    model.eval()
    loader_iterator = None
    try:
        with ExitStack() as stack:
            source = stack.enter_context(rasterio.open(image_path)) if not reuse_input else None
            labels = stack.enter_context(rasterio.open(label_path)) if label_path else None

            def completed_blocks(row_count):
                """已完成条带按 core 宽切块，避免显示、计量重新创建整景数组。"""
                for left in range(0, ds.width, ds.core):
                    width = min(ds.core, ds.width - left)
                    # 额外 halo 仅服务于边界计量，预测始终来自已融合缓冲。
                    # 未知滤波保留旧读取窗口，避免窗口相关滤波改变显示值/有效域。
                    context_halo = halo if boundary_context or not reuse_input else 0
                    x0, y0 = max(0, left - context_halo), max(0, start_row - context_halo)
                    x1 = min(ds.width, left + width + context_halo)
                    y1 = min(ds.height, start_row + row_count + context_halo)
                    window = Window(x0, y0, x1-x0, y1-y0)
                    if reuse_input:
                        # 独立副本保证调用方保留块或提前关闭迭代器时不引用滚动缓存。
                        image = cached_image[:, :row_count, left:left+width].copy()
                        valid = cached_valid[:row_count, left:left+width].copy()
                    else:
                        image, valid = read_filtered_window(source, bands, window, ds.filter)
                    # 与旧实现一致，在 CPU float32 上按原顺序归一化显示值。
                    if not inputs_normalized:
                        image -= means_np
                        image /= stds_np
                    image[:, ~valid] = 0
                    ys = slice(start_row-y0, start_row-y0+row_count)
                    xs = slice(left-x0, left-x0+width)
                    if not boundary_context:
                        image, valid = image[:, ys, xs], valid[ys, xs]
                        window = Window(left, start_row, width, row_count)
                        ys, xs = slice(0, row_count), slice(0, width)
                    truth = np.full(valid.shape, -1, np.int64)
                    if labels is not None:
                        values = labels.read(1, window=window, masked=True)
                        known = valid & ~np.ma.getmaskarray(values) & np.isin(values.data, (0, 1))
                        truth[known] = values.data[known]
                    denominator = totals[:row_count, left:left+width]
                    final_valid = valid[ys, xs] & (denominator > 0)
                    score = np.divide(sums[:row_count, left:left+width], denominator,
                                      out=np.zeros((row_count, width), np.float32),
                                      where=denominator > 0)
                    # 全部字段均为当前小块独立数组或其上下文视图，不引用滚动缓冲。
                    yield dict(window=(start_row, left, row_count, width), scores=score,
                               image=image[:, ys, xs], label=truth[ys, xs], valid_mask=final_valid,
                               label_context=truth, valid_context=valid, core_slices=(ys, xs),
                               scene_shape=(ds.height, ds.width))

            loader_iterator = iter(loader)
            for batch in loader_iterator:
                valid = batch['valid_mask'].to(device, non_blocking=True)
                image = batch['image'].to(device, non_blocking=True)
                if not inputs_normalized:
                    image = (image-means)/stds
                image = image.masked_fill(~valid[:, None], 0)
                with torch.inference_mode(), torch.autocast(device_type=device.type,
                                                           enabled=use_amp and device.type == 'cuda'):
                    prediction = model(image)
                    logits = prediction[-1] if isinstance(prediction, (tuple, list)) else prediction
                    scores = logits[:, 0].float() if logits.shape[1] == 1 else logits[:, 1].float()-logits[:, 0].float()
                    if not torch.isfinite(scores).all():
                        raise FloatingPointError(f'非有限模型输出：{image_path}')
                    # 先检查完整输出，再只传输模型核心区；halo 不参与融合。
                    scores = scores[:, halo:halo+ds.core, halo:halo+ds.core].contiguous().cpu().numpy()
                for index, (top, left, height, width) in enumerate(batch['window'].tolist()):
                    if top > start_row:
                        rows = top-start_row
                        yield from completed_blocks(rows)
                        # 上移仍待相邻窗口融合的部分，底部清零供新窗口累积。
                        sums[:-rows] = sums[rows:]
                        totals[:-rows] = totals[rows:]
                        sums[-rows:] = 0
                        totals[-rows:] = 0
                        if reuse_input:
                            cached_image[:, :-rows] = cached_image[:, rows:]
                            cached_valid[:-rows] = cached_valid[rows:]
                            cached_image[:, -rows:] = 0
                            cached_valid[-rows:] = False
                        start_row = top
                    ys, xs = slice(halo, halo+height), slice(halo, halo+width)
                    good = batch['valid_mask'][index, ys, xs].numpy()
                    weight = weights[:height, :width] * good
                    sums[:height, left:left+width] += scores[index, :height, :width] * weight
                    totals[:height, left:left+width] += weight
                    if reuse_input:
                        # 原始 CPU batch 像素只在输出时归一化一次，不复用 GPU 舍入值。
                        cached_image[:, :height, left:left+width] = batch['image'][index, :, ys, xs].numpy()
                        cached_valid[:height, left:left+width] = good
                del prediction, logits, scores, image, valid
            yield from completed_blocks(ds.height-start_row)
    finally:
        # generator.close()/预测异常也须回收 DataLoader 子进程及其栅格句柄。
        shutdown_workers = getattr(loader_iterator, '_shutdown_workers', None)
        if shutdown_workers is not None:
            shutdown_workers()
        ds.close()
        model.train(was_training)
