"""保留 S1S2-Water 的原下载命令，已有 .part 文件可继续使用。"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("data") / "download_s1s2_water.py"), run_name="__main__")
