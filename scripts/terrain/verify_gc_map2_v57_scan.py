"""用留存的x86-64指令重建扫描函数，独立核对C++移植；在Linux/WSL运行。

仅执行.omp_fn.6，GOMP来自本机libgomp。不运行完整gc_map2，不验证上游内部
网格或最终输出。构建写入build/gamma57_oracle，不覆盖生产核心。
"""

import hashlib
import json
from pathlib import Path
import re
import subprocess


def verify_gc_map2_v57_scan(evidence_dir, output_dir, cases=20000):
    """输入v5.7逆向目录、独立构建目录；重汇编后比较随机及边界剖面字节。"""
    evidence_dir = Path(evidence_dir)
    output_dir = Path(output_dir)
    manifest = json.loads((evidence_dir / 'data/model_evidence_manifest.json').read_text(encoding='utf-8'))
    target_sha = '6de93a4c12591753545aae9e9787f8041ba147e046ae03321737eb83c4262676'
    if manifest['binary_sha256'] != target_sha:
        raise ValueError('逆向资料不是指定gc_map2 v5.7目标')
    source = evidence_dir / 'disasm/omp_named.asm'
    assembly = ['.intel_syntax noprefix', '.text', '.global recovered_gamma57_scan', 'recovered_gamma57_scan:']
    instructions = 0
    for line in source.read_text(encoding='utf-8').splitlines():
        fields = line.split('\t')
        if len(fields) < 3:
            continue
        address = fields[0].strip().rstrip(':')
        if not re.fullmatch('[0-9a-f]+', address) or not 0xd080 <= int(address, 16) < 0xd410:
            continue
        instruction = fields[2].split('#')[0].strip()
        assembly.append(f'L_{address}:')
        # 不保留填充nop；跳转仍指向对应标签，计算指令与分支条件不变。
        if 'nop' in instruction:
            continue
        branch = re.fullmatch(r'(\w+)\s+([0-9a-f]+) <([^>]+)>', instruction)
        if branch:
            operation, destination, symbol = branch.groups()
            target = f'L_{destination}' if symbol.startswith('main._omp_fn.6') else symbol
            instruction = f'{operation} {target}'
        assembly.append('    ' + instruction)
        instructions += 1
    if not any('L_d3fc:' == line for line in assembly):
        raise ValueError('扫描函数反汇编不完整')
    assembly.append('.section .note.GNU-stack,"",@progbits')
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'recovered_scan.S').write_text('\n'.join(assembly) + '\n', encoding='utf-8')
    (output_dir / 'verify.cpp').write_text(HARNESS, encoding='utf-8')
    native = Path(__file__).resolve().parents[2] / 'src/sar_geometry/native'
    command = [
        'g++', '-std=c++20', '-O2', '-ffp-contract=off', '-fopenmp',
        '-I', str(native), str(output_dir / 'verify.cpp'),
        str(output_dir / 'recovered_scan.S'), str(native / 'gc_map2_v57.cpp'),
        '-o', str(output_dir / 'verify_scan'),
    ]
    print(f'[1/2] 重建{instructions}条计算/控制指令；独立编译', flush=True)
    subprocess.run(command, check=True)
    print(f'[2/2] 比较{cases}组剖面，覆盖正/反角度场、零值、等值和末端折返', flush=True)
    result = subprocess.run([str(output_dir / 'verify_scan'), str(cases)], capture_output=True, text=True)
    print(result.stdout, end='', flush=True)
    print(result.stderr, end='', flush=True)
    report = dict(
        target_version='gc_map2 v5.7 28-Jul-2023', target_binary_sha256=target_sha,
        evidence_assembly_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        scope='reassembled retained instructions of .omp_fn.6, finite float32 inputs only',
        full_gc_map2_executed=False, cases=cases, passed=result.returncode == 0,
        output=result.stdout, error=result.stderr,
    )
    (output_dir / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    result.check_returncode()
    return report


HARNESS = r'''
#include "gc_map2_v57.h"
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>
extern "C" void recovered_gamma57_scan(void*);
extern "C" void GOMP_parallel(void (*)(void*), void*, unsigned, unsigned);
struct ScanInput {
    float** angle;
    std::uint8_t** flags;
    float** range;
    int rows;
    int columns;
    int direction;
};
int main(int argc, char** argv) {
    int cases = argc > 1 ? std::stoi(argv[1]) : 20000;
    std::mt19937 random(5719);
    std::uint64_t pixels = 0;
    for (int test = 0; test < cases; ++test) {
        int rows = 1 + random() % 4;
        int columns = 1 + random() % 70;
        int sign = random() % 2 ? 1 : -1;
        std::vector<std::vector<float>> angle(rows, std::vector<float>(columns));
        auto range = angle;
        std::vector<std::vector<std::uint8_t>> flags(rows, std::vector<std::uint8_t>(columns, 0));
        std::vector<float*> angle_ptr(rows), range_ptr(rows);
        std::vector<std::uint8_t*> flag_ptr(rows);
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < columns; ++col) {
                // 小整数制造精确等值；连续随机浮点检查float32比较与double状态。
                float a = float(int(random() % 15) - 7);
                float r = float(1000 + random() % 12);
                if (test % 3 == 0) {
                    a += float(random() % 1000) / 3000;
                    r += float(random() % 1000) / 3000;
                }
                if (random() % 13 == 0) a = 0;
                if (random() % 17 == 0) r = 0;
                angle[row][col] = a;
                range[row][col] = r;
            }
            angle_ptr[row] = angle[row].data();
            range_ptr[row] = range[row].data();
            flag_ptr[row] = flags[row].data();
        }
        ScanInput input{angle_ptr.data(), flag_ptr.data(), range_ptr.data(), rows, columns, sign};
        GOMP_parallel(recovered_gamma57_scan, &input, test % 2 ? 1 : 4, 0);
        for (int row = 0; row < rows; ++row) {
            auto actual = sar::scan_gc_map2_v57_shadow_layover(angle[row], range[row], sign);
            if (actual != flags[row]) {
                std::cerr << "Mismatch test=" << test << " row=" << row << " sign=" << sign << '\n';
                for (int col = 0; col < columns; ++col)
                    std::cerr << col << ' ' << angle[row][col] << ' ' << range[row][col]
                              << ' ' << int(flags[row][col]) << ' ' << int(actual[col]) << '\n';
                return 1;
            }
            pixels += columns;
        }
    }
    std::cout << "PASS: " << cases << " profile batches; " << pixels << " samples; threads=1/4\n";
}
'''


if __name__ == '__main__':
    # WSL路径；只做秒级对照，不启动整景或批量水体处理。
    verify_gc_map2_v57_scan(
        '/mnt/e/Code_projects/SpaceTY_SAR_vs2022/reverse_engineering/gc_map2',
        Path(__file__).resolve().parents[2] / 'build/gamma57_oracle',
    )
