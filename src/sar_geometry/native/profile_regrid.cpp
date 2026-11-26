#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstddef>
// The quadratic cardinal B-spline and its derivative share the same 3x3 stencil.
static void weights(double u, double *w, double *d) {
    w[0] = 0.5 * (0.5-u)*(0.5-u);
    w[1] = 0.75-u*u;
    w[2] = 0.5 * (0.5+u)*(0.5+u);
    d[0] = u-0.5; d[1] = -2*u; d[2] = u+0.5;
}

static bool evaluate(const double *f, int h, int w, double r, double c,
                     double &value, double &dr, double &dc) {
    if (!(r >= 0 && c >= 0 && r <= h-1 && c <= w-1)) return false;
    const int i = int(std::floor(r+0.5)), j = int(std::floor(c+0.5));
    double wr[3], wc[3], dwr[3], dwc[3];
    weights(r-i, wr, dwr);
    weights(c-j, wc, dwc);
    value=dr=dc=0;
    for(int a=-1; a<=1; ++a) {
        int row = i+a;
        if (row < 0) row = -row;
        if (row >= h) row = 2*h-2-row;
        for (int b=-1; b<=1; ++b) {
            int col = j+b;
            if (col < 0) col = -col;
            if (col >= w) col = 2*w-2-col;
            const double z = f[static_cast<size_t>(row)*w+col];
            value += wr[a+1]*wc[b+1]*z;
            dr += dwr[a+1]*wc[b+1]*z;
            dc += wr[a+1]*dwc[b+1]*z;
        }
    }
    return std::isfinite(value);
}

#ifdef _WIN32
#define REGRID_EXPORT __declspec(dllexport)
#else
#define REGRID_EXPORT __attribute__((visibility("default")))
#endif
extern "C" REGRID_EXPORT
void gather(const double *az, const double *yb, const double *rg, const double *look,
            const uint8_t *valid,
            int h, int w, int r0, int c0, const double *p, const double *q,
            int a0, int k0, int nrows, int ncols, int full_rows, int full_cols,
            double ystart, double step, float *rangep, float *lookp) {
    for (int a=0; a<nrows; ++a) {
        const int target_a = a0+a;
        const double A = 2.*target_a/std::max(1, full_rows-1)-1.;
        for (int k=0; k<ncols; ++k) {
            const double B = 2.*(k0+k)/std::max(1,full_cols-1)-1.;
            const double b[] = {1,A,B,A*A,A*B,B*B,A*A*A,A*A*B,A*B*B,B*B*B};
            double r=-r0,c=-c0;
            for (int i=0; i<10; ++i) {r+=p[i]*b[i]; c+=q[i]*b[i];}
            const double target_y = ystart + step*(k0+k);
            for (int iter=0; iter<8; ++iter) {
                double v1,v2,ar,ac,yr,yc;
                if (!evaluate(az,h,w,r,c,v1,ar,ac) ||
                    !evaluate(yb,h,w,r,c,v2,yr,yc)) break;
                const double e1=target_a-v1, e2=target_y-v2;
                if (std::abs(e1)<1.e-4 && std::abs(e2)<1.e-4*step) {
                    if (!valid[static_cast<size_t>(std::floor(r+.5))*w+int(std::floor(c+.5))]) break;
                    double val_r,val_l,der1,der2;
                    if (evaluate(rg,h,w,r,c,val_r,der1,der2) &&
                        evaluate(look,h,w,r,c,val_l,der1,der2)) {
                        size_t i = static_cast<size_t>(a)*ncols+k;
                        rangep[i]=float(val_r); lookp[i]=float(val_l);
                    }
                    break;
                }
                const double det = ar*yc-ac*yr;
                if (!std::isfinite(det) || std::abs(det)<1.e-10) break;
                r+=(yc*e1-ac*e2)/det;
                c+=(ar*e2-yr*e1)/det;
            }
        }
    }
}
