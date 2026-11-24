#include "gc_map2_v57.h"
#include <cstddef>

namespace sar {
std::vector<std::uint8_t> scan_gc_map2_v57_shadow_layover(
    const std::vector<float>& angle, const std::vector<float>& range, int direction) {
    // 静态证据：SHA256 6de93a4c...2676，.omp_fn.6，0xd080–0xd3fc。
    // 保留float32比较、严格/非严格不等号、零值跳过和末端状态，不加epsilon。
    std::vector<std::uint8_t> flags(angle.size(), 0);
    bool shadow_active = false;
    double horizon = 0;
    for (std::size_t i = 1; i < angle.size(); ++i) {
        if (angle[i - 1] == 0 || angle[i] == 0) continue;
        float previous = angle[i - 1] * static_cast<float>(direction);
        float current = angle[i] * static_cast<float>(direction);
        if (previous > current && !shadow_active) {
            horizon = angle[i - 1];
            shadow_active = true;
        }
        if (shadow_active && direction * horizon > current) {
            flags[i] = 16;
        } else {
            flags[i] = 1;
            horizon = 0;
            shadow_active = false;
        }
    }

    bool fold_active = false;
    double fold_high = 0;
    double fold_low = 0;
    std::ptrdiff_t fold_start = 0;
    for (std::size_t i = 0; i + 1 < range.size(); ++i) {
        float previous = range[i];
        float current = range[i + 1];
        if (previous == 0 || current == 0) continue;

        // d390–d3bf：只有后一位置不是纯阴影16，才启动新的距离折返。
        if (previous >= current && !fold_active) {
            fold_high = 0;
            if (flags[i + 1] == 16) continue;
            fold_high = previous;
            fold_low = current;
            fold_start = static_cast<std::ptrdiff_t>(i);
            flags[i + 1] = 4;
            fold_active = true;
            continue;
        }

        // d3c8–d3fa：等斜距属于折返；阴影点可组合为20，但不降低折返下界。
        if (current <= fold_high && fold_active) {
            if (current < fold_low && flags[i + 1] != 16) fold_low = current;
            flags[i + 1] = flags[i + 1] == 16 ? 20 : 4;
            continue;
        }

        // d289–d2e1：脱离折返上界后，沿原地表次序向前补记同斜距分支。
        if (current > fold_high && fold_active) {
            for (auto j = fold_start; j >= 0 && range[j] >= fold_low; --j) {
                flags[j] = (flags[j] & ~4) == 16 ? 20 : 4;
            }
            fold_active = false;
        } else {
            fold_high = 0;
            fold_active = false;
        }
    }
    // 目标扫描到最后一个相邻对即退出，没有额外的末端折返回填。
    return flags;
}
}
