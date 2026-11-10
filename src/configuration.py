"""读取执行入口的 YAML；paths 中的绝对路径转换为 Path，不启动任何处理。"""
from pathlib import Path
import yaml


def load_yaml(path):
    """安全读取配置；不从环境变量或隐藏默认值补充机器路径。"""
    with Path(path).open(encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"YAML 顶层必须是参数分组：{path}")

    def path_value(value):
        if isinstance(value, dict):
            return {key: path_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [path_value(item) for item in value]
        return Path(value) if isinstance(value, str) else value

    config["paths"] = path_value(config["paths"])
    return config
