"""水体二分类像元指标：训练、独立评价和后处理共用同一统计口径。"""
from math import sqrt


def metrics_from_counts(counts):
    """输入 TN、FP、FN、TP；分母无定义时为 NaN，原 Precision=0 约定保持不变。"""
    tn, fp, fn, tp = map(int, counts)
    count = tn + fp + fn + tp
    nan = float("nan")
    water_iou = tp / (tp + fp + fn) if tp + fp + fn else nan
    background_iou = tn / (tn + fp + fn) if tn + fp + fn else nan
    recall = tp / (tp + fn) if tp + fn else nan
    specificity = tn / (tn + fp) if tn + fp else nan
    # Kappa 使用边际计数计算随机一致性；MCC 同时考虑四类计数。
    expected = (tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)
    kappa_denominator = count * count - expected
    mcc_denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return dict(
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else nan,
        iou=water_iou,
        precision=tp / (tp + fp) if tp + fp else 0.0,
        recall=recall,
        tn=tn, fp=fp, fn=fn, tp=tp, valid_pixels=count,
        true_water_ratio=(tp + fn) / count if count else nan,
        pred_water_ratio=(tp + fp) / count if count else nan,
        accuracy=(tp + tn) / count if count else nan,
        specificity=specificity,
        false_positive_rate=fp / (fp + tn) if fp + tn else nan,
        false_negative_rate=fn / (fn + tp) if fn + tp else nan,
        false_discovery_rate=fp / (fp + tp) if fp + tp else nan,
        background_iou=background_iou,
        # 严格两类平均：任何一类无定义时保留 NaN，不悄悄变成单类平均。
        mean_iou=(water_iou + background_iou) / 2,
        balanced_accuracy=(recall + specificity) / 2,
        kappa=(count * (tp + tn) - expected) / kappa_denominator if kappa_denominator else nan,
        mcc=(tp * tn - fp * fn) / sqrt(mcc_denominator) if mcc_denominator else nan,
    )
