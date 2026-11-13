#pragma once
#include <array>
#include <vector>
namespace sar {
// 地形支持域之外采用明确高程包络。角度弧度，范围米；未知255，不视为无叠掩/阴影。
std::vector<std::array<double,2>> visibility_bounded(const std::vector<std::array<double,4>>& points,
    double preceding_angle_upper,double preceding_range_upper,double following_range_lower,
    double angle_epsilon,double range_epsilon);
// 输入按近至远地表顺序：[s,斜距,向上有向仰角(rad),valid]。
// 仅用于已建立的同一观测剖面；不负责从DEM构造剖面。
using ProfilePoint=std::array<double,4>;
using Visibility=std::array<double,2>; // shadow, layover；0/1/255未知
std::vector<Visibility> visibility(const std::vector<ProfilePoint>& points,
                                 bool support_complete,double angle_epsilon,double range_epsilon);
}
