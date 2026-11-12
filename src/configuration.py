"""读取执行入口的 YAML；可用 DATA_ROOT 统一解析数据目录，不启动处理。"""
from pathlib import Path
import yaml


def load_yaml(path):
    """安全读取配置；不从环境变量或隐藏默认值补充机器路径。"""
    with Path(path).open(encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"YAML 顶层必须是参数分组：{path}")

    # DATA_ROOT 是唯一数据位置；其余相对路径以它为基准，与终端目录无关。
    # 未声明 DATA_ROOT 的旧配置保持原解析规则；软件/仓库绝对路径不受影响。
    root = config['paths'].get('DATA_ROOT')
    if root is not None:
        root = Path(root)
        if not root.is_absolute():
            root = Path(path).resolve().parent / root
        root = root.resolve()

    def path_value(value):
        if isinstance(value, dict):
            return {key: path_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [path_value(item) for item in value]
        if isinstance(value, str):
            value = Path(value)
            return (root / value).resolve() if root is not None and not value.is_absolute() else value
        return value

    config["paths"] = path_value(config["paths"])
    if root is not None:
        config['paths']['DATA_ROOT'] = root
    return config
