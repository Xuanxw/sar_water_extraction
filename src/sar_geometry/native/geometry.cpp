#include "geometry.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace sar {
Vec add(Vec a, Vec b) { return {a[0]+b[0],a[1]+b[1],a[2]+b[2]}; }
Vec sub(Vec a, Vec b) { return {a[0]-b[0],a[1]-b[1],a[2]-b[2]}; }
Vec mul(Vec a, double s) { return {a[0]*s,a[1]*s,a[2]*s}; }
double dot(Vec a, Vec b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
Vec cross(Vec a, Vec b) { return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]}; }
double norm(Vec a) { return std::sqrt(dot(a,a)); }

State interpolate_orbit(const std::vector<State>& orbit, double t) {
    if (t<orbit.front().time || t>orbit.back().time) throw std::out_of_range("orbit time");
    auto it=std::upper_bound(orbit.begin(),orbit.end(),t,[](double x,const State& s){return x<s.time;});
    std::size_t i=it==orbit.end()?orbit.size()-2:static_cast<std::size_t>(it-orbit.begin()-1);
    const auto& a=orbit[i]; const auto& b=orbit[i+1];
    double dt=b.time-a.time,u=(t-a.time)/dt,u2=u*u,u3=u2*u;
    State out{t,{},{}};
    // 分段三次Hermite同时使用位置与速度；速度取同一位置曲线的导数。
    for (int k=0;k<3;++k) {
        out.position[k]=(2*u3-3*u2+1)*a.position[k]+(u3-2*u2+u)*dt*a.velocity[k]
                       +(-2*u3+3*u2)*b.position[k]+(u3-u2)*dt*b.velocity[k];
        out.velocity[k]=(6*u2-6*u)/dt*a.position[k]+(3*u2-4*u+1)*a.velocity[k]
                       +(-6*u2+6*u)/dt*b.position[k]+(3*u2-2*u)*b.velocity[k];
    }
    return out;
}

static double polynomial(const DopplerRecord& d,double range) {
    double x=2*range/299792458.0-d.tau0, y=0;
    for (auto i=d.coefficients.rbegin();i!=d.coefficients.rend();++i) y=y*x+*i;
    return y;
}
static double doppler(const Model& m,double range,double time) {
    if (m.doppler_mode==0) return 0;
    const auto& d=m.doppler;
    if (time<d.front().time || time>d.back().time) throw std::out_of_range("Doppler time");
    auto it=std::upper_bound(d.begin(),d.end(),time,[](double t,const DopplerRecord& a){return t<a.time;});
    if (it==d.end()) return m.doppler_sign*polynomial(d.back(),range);
    if (it==d.begin()) return m.doppler_sign*polynomial(d.front(),range);
    const auto& a=*(it-1); const auto& b=*it; double w=(time-a.time)/(b.time-a.time);
    // 每个估计保留自己的距离原点；在时刻之间插值Hz，不直接混合不同原点系数。
    return m.doppler_sign*((1-w)*polynomial(a,range)+w*polynomial(b,range));
}

Result solve(const Model& m,Vec p,Vec n) {
    double nan=std::numeric_limits<double>::quiet_NaN();
    Result out{nan,nan,nan,nan,0,InvalidInput,nan,nan};
    if (!std::isfinite(norm(p)) || norm(p)<1) return out;
    double t=m.time0+0.5*(m.rows-1)*m.line_time;
    for (int iteration=1;iteration<=static_cast<int>(m.max_iterations);++iteration) {
        out[2]=t;out[4]=iteration;
        if (t<m.orbit.front().time || t>m.orbit.back().time) {out[5]=OrbitBounds;return out;}
        State s=interpolate_orbit(m.orbit,t); Vec l=sub(p,s.position);
        double r=norm(l),speed2=dot(s.velocity,s.velocity),fd;
        if (!std::isfinite(r) || r<=0 || !std::isfinite(speed2) || speed2<=0) return out;
        try { fd=doppler(m,r,t); } catch(const std::out_of_range&) {out[5]=DopplerBounds;return out;}
        double g=dot(s.velocity,l)-0.5*m.wavelength*r*fd,step=g/speed2;
        out[0]=(r-m.range0)/m.range_spacing;out[1]=(t-m.time0)/m.line_time;
        out[3]=2*g/(m.wavelength*r);out[7]=std::abs(step)/m.line_time;
        if (!std::isfinite(out[3]) || !std::isfinite(step)) return out;
        if (std::abs(out[3])<=m.tolerance_hz && out[7]<=m.tolerance_pixels) {
            // cross(V,S)指向右视侧；采用地心向量定义侧别，不把角度或入射角代替视侧。
            Vec right=cross(s.velocity,s.position);double side=dot(l,right)/norm(right);
            if (!std::isfinite(side) || side*m.look_side<=0) {out[5]=WrongSide;return out;}
            out[5]=(out[0]>=0 && out[0]<=m.columns-1 && out[1]>=0 && out[1]<=m.rows-1)?Valid:OutsideRadar;
            if (std::isfinite(norm(n)) && norm(n)>0)
                out[6]=std::acos(std::clamp(dot(n,mul(l,-1))/(norm(n)*r),-1.0,1.0))*180/std::numbers::pi;
            return out;
        }
        // 大步长限幅不改变最终成功条件，也不将越界解夹到合法时间上。
        t+=std::clamp(step,-m.max_step_seconds,m.max_step_seconds);
    }
    out[5]=NotConverged;return out;
}
}
