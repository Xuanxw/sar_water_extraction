#pragma once
#include <cstdint>
#include <vector>

namespace sar {
// gc_map2 v5.7 .omp_fn.6 的有限float32输入扫描；不是完整gc_map2链路。
// 输入保持内部横向顺序，0沿用目标程序的缺测哨兵；direction是角度场方向±1，
// 不能直接以左右视或DEM行序代替。输出内部字节0/1/4/16/20。
std::vector<std::uint8_t> scan_gc_map2_v57_shadow_layover(
    const std::vector<float>& angle, const std::vector<float>& range, int direction);
}
