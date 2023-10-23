#pragma once
#include <algorithm>

namespace adelie_core {
namespace optimization {

/*
 * Brent's Method: https://mmas.github.io/brent-julia
 */
template <class F, class ValueType, class ExtraCheckType>
inline
void brent(
    F f,
    ValueType x_tol,
    ValueType y_tol,
    size_t max_iters,
    ValueType a,
    ValueType c,
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
        std::swap(fa, fb);
    }
    value_t fc = (c == a) ? fa : f(c);
    value_t d = c;
    bool do_bisect = true;
    
    iters = 0;
    for (; iters < max_iters; ++iters) {
        const auto extra_check_state = extra_check_f(a, fa, b, fb);
        if (std::get<0>(extra_check_state)) {
            sol = std::get<1>(extra_check_state);
            return;
        }

        if ((std::abs(b-a) <= x_tol) || (std::abs(fb) <= 2*y_tol)) {
            sol = b;
            return;
        }

        value_t s = 0;
        if (std::abs(fa-fc) > y_tol && std::abs(fb-fc) > y_tol) {
            s = (
                a * fb * fc / ((fa-fb) * (fa-fc)) +
                b * fa * fc / ((fb-fa) * (fb-fc)) +
                c * fa * fb / ((fc-fa) * (fc-fb))
            );
        }
        else {
            s = b - fb * (b-a) / (fb - fa);
        }
        
        const auto delta = std::abs(2 * x_tol * std::abs(b));
        const auto min1 = std::abs(s-b);
        const auto min2 = std::abs(b-c);
        const auto min3 = std::abs(c-d);
        if ((s < 0.25*(3*a+b) && s > b) ||
            (do_bisect && min1 >= min2 * 0.5) ||
            (!do_bisect && min1 >= min3 * 0.5) ||
            (do_bisect && min2 < delta) ||
            (!do_bisect && min3 < delta)) {
            s = (a + b) * 0.5;
            do_bisect = true;
        } else {
            do_bisect = false;
        }
        
        value_t fs = f(s);
        if (std::abs(fs) < y_tol) {
            sol = s;
            return;
        }

        d = c;
        c = b;
        if (fs * fa < 0) {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }

        if (std::abs(fa) < std::abs(fb)) {
            std::swap(a, b);
            std::swap(fa, fb);
        }
    }
}

} // namespace optimization 
} // namespace adelie_core