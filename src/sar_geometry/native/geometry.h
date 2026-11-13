#pragma once
#include <array>
#include <vector>
#include <cstddef>

namespace sar {
using Vec = std::array<double, 3>;
Vec add(Vec a, Vec b);
Vec sub(Vec a, Vec b);
Vec mul(Vec a, double s);
double dot(Vec a, Vec b);
Vec cross(Vec a, Vec b);
double norm(Vec a);
struct State { double time; Vec position, velocity; };
struct DopplerRecord { double time, tau0; std::vector<double> coefficients; };
struct Model {
    // 同一连续时间轴；距离为米，Doppler多项式自变量为双程时间差（秒）。
    double wavelength, range0, range_spacing, time0, line_time;
    double columns, rows, look_side, doppler_mode, doppler_sign;
    double max_iterations, tolerance_hz, tolerance_pixels, max_step_seconds;
    std::vector<State> orbit;
    std::vector<DopplerRecord> doppler;
};
enum Status { Valid=0, InvalidInput=1, OrbitBounds=2, NotConverged=3,
              WrongSide=4, DopplerBounds=5, OutsideRadar=6 };
// 输出：距离像元、方位像元、时刻、残差Hz、迭代数、状态、局部入射角度、时间步长像元。
using Result = std::array<double, 8>;
State interpolate_orbit(const std::vector<State>& orbit, double time);
Result solve(const Model& model, Vec point, Vec normal);
}
