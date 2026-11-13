#include "visibility.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
namespace sar {
std::vector<Visibility> visibility_bounded(const std::vector<ProfilePoint>& p,double upper,double near_max,double far_min,double ae,double re) {
    auto out=visibility(p,true,ae,re);
    double horizon=-std::numeric_limits<double>::infinity();bool gap=false;
    for(std::size_t i=0;i<p.size();++i) {
        bool valid=p[i][3]==1 && std::isfinite(p[i][1]) && std::isfinite(p[i][2]) && p[i][1]>0;
        if(!valid){out[i]={255,255};gap=true;continue;}
        // 已观测前山遮挡可确定为1；可见则须同时越过未观测近侧地形的上界。
        out[i][0]=p[i][2]<horizon-ae?1:(!gap && p[i][2]>std::max(horizon,upper)+ae?0:255);
        horizon=std::max(horizon,p[i][2]);
        // 地表支持域外仍可能有同斜距分支。只有已证明的折返1不受此未知范围取消。
        if(out[i][1]!=1 && (p[i][1]<=near_max+re || p[i][1]>=far_min-re))out[i][1]=255;
    }
    return out;
}
std::vector<Visibility> visibility(const std::vector<ProfilePoint>& p,bool complete,double ae,double re) {
    std::vector<Visibility> out(p.size(),{255,255});
    if (p.empty()) return out;
    bool all_valid=complete;
    double horizon=-std::numeric_limits<double>::infinity();
    bool known=complete;
    for (std::size_t i=0;i<p.size();++i) {
        if (!std::isfinite(p[i][0]) || (i && p[i][0]<=p[i-1][0]))
            throw std::runtime_error("profile s must be finite and strictly increasing");
        if (p[i][3]!=1 || !std::isfinite(p[i][1]) || !std::isfinite(p[i][2]) || p[i][1]<=0) {
            known=false;all_valid=false;continue;
        }
        if (known) {
            // 临界共线视为未知；首个支持点在明确complete时可见。
            out[i][0]=p[i][2]<horizon-ae?1:(p[i][2]>horizon+ae?0:255);
            horizon=std::max(horizon,p[i][2]);
        }
    }
    // 缺失剖面可能隐藏新的距离折返；保守保留整条叠掩未知，不擅自跨洞插值。
    if (!all_valid) return out;
    std::vector<std::pair<double,double>> folds;
    std::vector<double> critical;
    for (std::size_t i=1;i<p.size();++i) {
        double a=p[i-1][1],b=p[i][1];
        if (b<a-re) folds.emplace_back(b,a);
        else if (std::abs(a-b)<=re) {critical.push_back(a);critical.push_back(b);}
    }
    std::sort(folds.begin(),folds.end());
    std::vector<std::pair<double,double>> intervals;
    for (auto f:folds) {
        if (intervals.empty() || f.first>intervals.back().second+re) intervals.push_back(f);
        else intervals.back().second=std::max(intervals.back().second,f.second);
    }
    std::sort(critical.begin(),critical.end());
    for (std::size_t i=0;i<p.size();++i) {
        double r=p[i][1];out[i][1]=0;
        auto it=std::upper_bound(intervals.begin(),intervals.end(),r+re,
            [](double x,const auto& f){return x<f.first;});
        if (it!=intervals.begin()) {
            --it;
            if (r>=it->first-re && r<=it->second+re)
                out[i][1]=(r>it->first+re && r<it->second-re)?1:255;
        }
        auto c=std::lower_bound(critical.begin(),critical.end(),r-re);
        if (c!=critical.end() && *c<=r+re && out[i][1]!=1) out[i][1]=255;
    }
    return out;
}
}
