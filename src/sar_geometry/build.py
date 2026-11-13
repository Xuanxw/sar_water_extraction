"""构建独立C++核心；构建产物放仓库build，不修改SPHERE工程。"""
from pathlib import Path
import os
import shutil
import subprocess

SOURCE_DIR = Path(__file__).resolve().parent/'native'
BUILD_DIR = Path(__file__).resolve().parents[2]/'build/sar_geometry'


def main():
    command=['cmake','-S',str(SOURCE_DIR),'-B',str(BUILD_DIR),'-DCMAKE_BUILD_TYPE=Release']
    if not (BUILD_DIR/'CMakeCache.txt').exists() and shutil.which('ninja'):
        command+=['-G','Ninja']
        if os.name=='nt' and shutil.which('g++'):
            command+=['-DCMAKE_CXX_COMPILER='+shutil.which('g++')]
    subprocess.run(command,check=True)
    subprocess.run(['cmake','--build',str(BUILD_DIR),'--config','Release'],check=True)
    print(f'构建完成：{BUILD_DIR}；请将实际可执行文件位置填入sar_geometry.yaml的CORE_PATH。')


if __name__=='__main__':
    main()
