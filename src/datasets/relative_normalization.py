"""训练集固定分位数的 VV 相对缩放；输入是存储值，不赋予物理 dB 含义。"""

import math
import hashlib
import json
from numbers import Integral, Real

import numpy as np


REPRESENTATION = "vv_train_p02_p98_dem_m"
LOG_REPRESENTATION = "vv_log_train_p02_p98_dem_m"
RELATIVE_REPRESENTATIONS = (REPRESENTATION, LOG_REPRESENTATION)
INPUT_DOMAIN = "dfc_stored_values_unknown_physical_scale"
LOG_POWER_FLOOR = 1e-6


def relative_log10_power(values):
    """对有效非负值计算 10 log10(max(x, 1e-6) / 1)，返回 float64。

    固定参考值和下限均以输入单位计。零是有效观测，映射到 -60；调用方负责掩膜。
    对未校正的功率只表示相对功率，对物理单位未知的 DFC 存储值只表示对数对比度。
    """
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("相对对数变换要求有效输入为有限非负值")
    return 10.0 * np.log10(np.maximum(values, LOG_POWER_FLOOR))


def _validated_bands(bands):
    bands = list(bands) if isinstance(bands, (list, tuple)) else []
    if (len(bands) != 2 or bands[0] == bands[1]
            or any(isinstance(b, bool) or not isinstance(b, Integral) or b < 1 for b in bands)):
        raise ValueError("训练文件签名需要两个不同的正整数波段，依次为 VV 和 DEM")
    return [int(b) for b in bands]


def training_files_signature(names, bands):
    """对训练样本相对名及 VV/DEM 波段顺序签名，不包含机器相关根目录。"""
    names = list(names)
    if not names or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("训练文件签名需要非空且不重复的样本名")
    names = sorted(names)
    bands = _validated_bands(bands)
    content = json.dumps(dict(files=names, bands=bands), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return dict(training_samples=len(names), train_files_sha256=hashlib.sha256(content).hexdigest(), bands=bands)


def relative_normalization_protocol(representation):
    """生成 VV 训练分位数的固定协议字段，供训练写入与后续读取时独立核对。"""
    if representation not in RELATIVE_REPRESENTATIONS:
        raise ValueError(f"未知训练分位数表示：{representation}")
    constants = {
        "version": 1,
        "method": "train_p02_p98",
        "representation": representation,
        "input_domain": INPUT_DOMAIN,
        "fit_scope": "all_valid_training_pixels",
        "percentiles": [2, 98],
        "range": [0.0, 1.0],
    }
    if representation == LOG_REPRESENTATION:
        constants.update(method="log_train_p02_p98",
                         percentile_domain="relative_log_dfc_stored_values",
                         log_formula="10*log10(max(x, floor)/reference)",
                         floor=LOG_POWER_FLOOR, reference=1.0)
    return constants


def validate_relative_normalization(parameters, expected_representation=None):
    """核对快照中的协议、拟合区间与预期输入表示；不重新估计训练集分位数。"""
    if not isinstance(parameters, dict):
        raise ValueError("需要已拟合的训练集 VV P02/P98 参数")
    representation = parameters.get("representation")
    if expected_representation is not None and representation != expected_representation:
        raise ValueError("训练分位数参数与当前 VV 输入表示不匹配")
    constants = relative_normalization_protocol(representation)
    required = set(constants) | {"p02", "p98", "fit_pixels", "training_samples", "train_files_sha256", "bands"}
    if set(parameters) != required or any(parameters.get(key) != value for key, value in constants.items()):
        raise ValueError("训练集 VV P02/P98 参数协议不匹配")
    if representation == LOG_REPRESENTATION and isinstance(parameters["reference"], bool):
        raise ValueError("对数参考值必须为数值")
    for key in ("p02", "p98"):
        value = parameters[key]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"归一化 {key} 必须是有限数")
    if parameters["p98"] <= parameters["p02"]:
        raise ValueError("训练集 VV P98 必须大于 P02；退化输入不能自动归一化")
    if representation == LOG_REPRESENTATION and parameters["p02"] < 10.0 * math.log10(LOG_POWER_FLOOR):
        raise ValueError("对数分位数不能低于有效零值映射的下限")
    if not math.isfinite(float(parameters["p98"]) - float(parameters["p02"])):
        raise ValueError("训练集 VV 分位区间宽度必须为有限数")
    for key in ("fit_pixels", "training_samples"):
        value = parameters[key]
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"归一化 {key} 必须是正整数")
    signature = parameters["train_files_sha256"]
    if not isinstance(signature, str) or len(signature) != 64 or any(c not in "0123456789abcdef" for c in signature):
        raise ValueError("归一化参数缺少有效的训练文件签名")
    bands = _validated_bands(parameters["bands"])
    result = dict(constants)
    result.update(p02=float(parameters["p02"]), p98=float(parameters["p98"]),
                  fit_pixels=int(parameters["fit_pixels"]), training_samples=int(parameters["training_samples"]),
                  train_files_sha256=signature, bands=bands)
    return result


def apply_vv_train_p02_p98_dem_m(image, valid, parameters):
    """共同有效域上固定缩放 VV，保留 DEM 米值；所有窗口复用训练集上下限。

    输入 image 为 [2,H,W] 的 VV 存储值和米制 DEM，valid 为同网格掩膜。
    输出 float32，VV 截断到 [0,1]，DEM 不缩放，掩膜外两个通道均填零。
    这不是辐射校正；禁止把不同存储单位直接套用同一份统计参数。
    """
    parameters = validate_relative_normalization(parameters)
    image = np.asarray(image)
    valid = np.asarray(valid, dtype=bool)
    if image.ndim != 3 or image.shape[0] != 2 or valid.shape != image.shape[1:]:
        raise ValueError("VV + DEM 需要同网格的 [2,H,W] 影像和 [H,W] 掩膜")
    if not np.isfinite(image[:, valid]).all():
        raise ValueError("有效掩膜内的 VV 和 DEM 必须为有限数")
    result = np.zeros(image.shape, dtype=np.float32)
    # 只计算有效像元，避免无效 NaN/Inf 参与相减；固定上下限不依赖窗口内容。
    values = image[0, valid].astype(np.float64)
    if parameters["representation"] == LOG_REPRESENTATION:
        values = relative_log10_power(values)
    scaled = (values - parameters["p02"]) / (parameters["p98"] - parameters["p02"])
    result[0, valid] = np.clip(scaled, 0.0, 1.0)
    result[1, valid] = image[1, valid]
    return result
