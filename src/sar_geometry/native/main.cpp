#include "geometry.h"
#include "visibility.h"
#include <bit>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

// 数值协议仅位于此I/O层：小端float64，前两个uint64为协议版本与样本数。
template<class T> T read_value() {
    T v{};if (!std::cin.read(reinterpret_cast<char*>(&v),sizeof(v))) throw std::runtime_error("truncated input");return v;
}
double number() {double v=read_value<double>();if (!std::isfinite(v)) throw std::runtime_error("nonfinite parameter");return v;}
std::size_t count(std::size_t max) {auto n=read_value<std::uint64_t>();if (n>max) throw std::runtime_error("excessive record count");return static_cast<std::size_t>(n);}
void write_values(const double* values,std::size_t size) {
    if (!std::cout.write(reinterpret_cast<const char*>(values),static_cast<std::streamsize>(size*sizeof(double)))) throw std::runtime_error("output failed");
}
int main(int argc,char** argv) {
    try {
        // 管道按大块传输；关闭stdio同步，避免逐数值I/O锁成为计算瓶颈。
        std::ios::sync_with_stdio(false);std::cin.tie(nullptr);
        if (argc!=2) throw std::runtime_error("usage: sar_geometry_core solve|visibility|version");
        std::string command=argv[1];
        if (command=="version") {std::cout<<"sar_geometry_core 0.2 protocol=1 openmp\n";return 0;}
        if constexpr (std::endian::native!=std::endian::little) throw std::runtime_error("little endian host required");
#ifdef _WIN32
        _setmode(_fileno(stdin),_O_BINARY);_setmode(_fileno(stdout),_O_BINARY);
#endif
        if (read_value<std::uint64_t>()!=1) throw std::runtime_error("unsupported protocol");
        auto n=count(10000000);
        if (command=="states") {
            // 轨道插值只在核心实现；Python批量请求构造剖面所需的卫星位置/速度。
            auto no=count(100000);if (no<2) throw std::runtime_error("two states required");
            std::vector<sar::State> orbit(no);
            for (std::size_t i=0;i<no;++i) {
                auto& s=orbit[i];s.time=number();for(auto& x:s.position)x=number();for(auto& x:s.velocity)x=number();
                if(i && s.time<=orbit[i-1].time)throw std::runtime_error("orbit times not increasing");
            }
            for(std::size_t i=0;i<n;++i) {
                auto s=sar::interpolate_orbit(orbit,number());
                write_values(s.position.data(),3);write_values(s.velocity.data(),3);
            }
        } else if (command=="solve") {
            sar::Model m{};
            m.wavelength=number();m.range0=number();m.range_spacing=number();m.time0=number();m.line_time=number();
            m.columns=number();m.rows=number();m.look_side=number();m.doppler_mode=number();m.doppler_sign=number();
            m.max_iterations=number();m.tolerance_hz=number();m.tolerance_pixels=number();m.max_step_seconds=number();
            if (m.wavelength<=0 || m.range_spacing<=0 || m.line_time<=0 || m.columns<1 || m.rows<1
                || std::abs(m.look_side)!=1 || std::abs(m.doppler_sign)!=1 || (m.doppler_mode!=0 && m.doppler_mode!=1)
                || m.max_iterations<1 || m.max_iterations>10000 || m.tolerance_hz<=0 || m.tolerance_pixels<=0 || m.max_step_seconds<=0)
                throw std::runtime_error("invalid solver parameters");
            auto no=count(100000);if (no<2) throw std::runtime_error("at least two orbit states required");
            for (std::size_t i=0;i<no;++i) {
                sar::State s{};s.time=number();for (auto& v:s.position) v=number();for (auto& v:s.velocity) v=number();
                if (i && s.time<=m.orbit.back().time) throw std::runtime_error("orbit times not increasing");
                m.orbit.push_back(s);
            }
            auto nd=count(100000),nc=count(20);
            if (m.doppler_mode==1 && (nd<2 || nc<1)) throw std::runtime_error("tabulated Doppler needs two estimates");
            for (std::size_t i=0;i<nd;++i) {
                sar::DopplerRecord d{};d.time=number();d.tau0=number();
                for (std::size_t j=0;j<nc;++j) d.coefficients.push_back(number());
                if (i && d.time<=m.doppler.back().time) throw std::runtime_error("Doppler times not increasing");
                m.doppler.push_back(d);
            }
            std::vector<std::array<double,6>> input(n);std::vector<sar::Result> result(n);
            for(auto& p:input)for(auto& x:p)x=read_value<double>();
            // 点之间没有迭代状态依赖，可以并行；单点的时间求根仍按顺序执行。
            #pragma omp parallel for schedule(static)
            for(std::int64_t i=0;i<static_cast<std::int64_t>(n);++i) {
                const auto& p=input[i];result[i]=sar::solve(m,{p[0],p[1],p[2]},{p[3],p[4],p[5]});
            }
            for(const auto& r:result)write_values(r.data(),r.size());
        } else if (command=="profiles") {
            auto columns=count(1000000);
            if(!columns || n%columns)throw std::runtime_error("invalid profile shape");
            double ae=number(),re=number();
            if(ae<0 || re<0)throw std::runtime_error("negative tolerance");
            auto rows=n/columns;std::vector<std::array<double,3>> bounds(rows);
            for(auto& row:bounds)for(auto& x:row)x=number();
            std::vector<std::vector<sar::ProfilePoint>> input(rows,std::vector<sar::ProfilePoint>(columns));
            for(auto& row:input) {
                for(auto& p:row)for(auto& x:p)x=read_value<double>();
                for(std::size_t i=0;i<columns;++i)
                    if(!std::isfinite(row[i][0]) || (i && row[i][0]<=row[i-1][0]))
                        throw std::runtime_error("profile coordinate must increase");
            }
            std::vector<std::vector<sar::Visibility>> output(rows);
            // 独立方位剖面并行；每一条必须整体扫描，不能沿距离拆块后重置包络。
            #pragma omp parallel for schedule(static)
            for(std::int64_t j=0;j<static_cast<std::int64_t>(rows);++j) {
                auto& row=input[j];std::size_t first=0,last=columns;
                while(first<last && row[first][3]!=1)++first;
                while(last>first && row[last-1][3]!=1)--last;
                output[j].assign(columns,{255,255});
                if(first<last) {
                    // 仅剥离外部缺测首尾；内部空洞仍传播未知，不跨洞拼接。
                    std::vector<sar::ProfilePoint> supported(row.begin()+first,row.begin()+last);
                    auto value=sar::visibility_bounded(supported,bounds[j][0],bounds[j][1],bounds[j][2],ae,re);
                    std::copy(value.begin(),value.end(),output[j].begin()+first);
                }
            }
            for(const auto& row:output)for(const auto& r:row)write_values(r.data(),r.size());
        } else if (command=="visibility") {
            bool complete=number()==1;double ae=number(),re=number();
            if (ae<0 || re<0) throw std::runtime_error("negative profile tolerance");
            std::vector<sar::ProfilePoint> points(n);
            for (auto& p:points) for (auto& v:p) v=read_value<double>();
            for (const auto& r:sar::visibility(points,complete,ae,re)) write_values(r.data(),r.size());
        } else throw std::runtime_error("unknown command");
        if (std::cin.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("unexpected trailing input");
        return 0;
    } catch (const std::exception& e) {std::cerr<<"sar_geometry_core: "<<e.what()<<'\n';return 1;}
}
