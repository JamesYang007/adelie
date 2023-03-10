#pragma once
#include <algorithm>

namespace glstudy {
    
/*
 * Brent's Method: https://users.wpi.edu/~walker/MA3257/HANDOUTS/brents_algm.pdf
 */
template <class F, class ValueType, class ExtraCheckType>
inline
void brent(
    F f,
    ValueType tol,
    size_t max_iters,
    ValueType a,
    ValueType b,
    ExtraCheckType extra_check_f,
    ValueType& sol,
    size_t& iters
)
{
    using value_t = ValueType;

    value_t fa = f(a);
    value_t fb = f(b);

    const bool do_switch = std::abs(fa) < std::abs(fb);
    if (do_switch) {
        std::swap(a, b);
    }
    value_t c = a;
    value_t fc = fa;
    
    iters = 0;
    for (; iters < max_iters; ++iters) {
        const auto extra_check_pack = extra_check_f(a, fa, b, fb);
        if (std::get<0>(extra_check_pack)) {
            sol = std::get<1>(extra_check_pack);
            return;
        }

        if ((std::abs(b-c) <= tol) || (std::abs(fb) <= tol)) {
            sol = b;
            return;
        }

        value_t s = 0;
        if ((fa != fc) && (fb != fc) && ((b-a) < tol*b)) {
            s = (
                (fb * fa) / ((fc - fb) * (fc - fa) * c)
                + (fc * fa) / ((fb-fc) * (fb-fa) * b)
                + (fc * fb) / ((fa-fc) * (fa-fb) * a)
            );
        }
        else {
            s = a + fa * (a-b) / (fb - fa);
        }
        
        if ((s < a) || (s > b)) {
            s = (a + b) * 0.5;
        }
        
        value_t fs = f(s);
        c = b;
        if (fs * fb < 0) {
            a = b;
            fa = fb;
        } else {
            fa *= 0.5;
        }
        b = s;
        fb = fs;
    }
}

} // namespace glstudy