#pragma once
#include <cmath>
#include <algorithm>

namespace adelie_core {
namespace optimization {

template <class VType>
auto _median(const VType& v)
{
    const auto K = v.size();
    return (K % 2 == 0) ? (0.5 * (v[K/2-1] + v[K/2])) : v[K/2];
}

template <class KnotsType, class ValueType>
ValueType symmetric_penalty(
    const KnotsType& knots,
    ValueType alpha
)
{
    const auto K = knots.size();

    if (K <= 0) return 0.0;

    const auto med = _median(knots);
    if (alpha >= 1) return med;

    const auto mean = knots.mean();
    if (alpha <= 0) return mean;

    const auto a_left = std::min(med, mean);
    const auto a_right = std::max(med, mean);
    if (a_right <= a_left) return a_left;

    const auto a_begin = std::lower_bound(
        knots.data(), 
        knots.data() + knots.size(),
        a_left
    ) - knots.data();
    const auto a_end = std::lower_bound(
        knots.data() + a_begin,
        knots.data() + knots.size(),
        a_right
    ) - knots.data();

    const auto sq_mean = knots.square().mean();
    const auto alpha_ratio = alpha / (1-alpha);

    const auto quad_min = [&](
        ValueType i,
        ValueType lower,
        ValueType upper,
        ValueType partial_mean
    ) 
    {
        const auto t_star = mean + alpha_ratio * (1 - 2*i/K);
        const auto argmin = (
            (t_star <= lower) ? lower : (
                (t_star <= upper) ? t_star : upper
            )
        );
        const auto f_min = (
            argmin * (argmin - 2 * t_star) + sq_mean + 2 * alpha_ratio * partial_mean
        );
        return std::make_pair(argmin, f_min);
    };

    ValueType partial_mean = mean - 2 * knots.head(a_begin).sum() / K;
    ValueType argmin;
    ValueType f_min;
    std::tie(argmin, f_min) = quad_min(
        a_begin, a_left, knots[a_begin], partial_mean
    );

    for (int i = a_begin+1; i < a_end; ++i) {
        partial_mean -= 2 * knots[i-1] / K;
        ValueType curr_argmin;
        ValueType curr_f_min;
        std::tie(curr_argmin, curr_f_min) = quad_min(
            i, knots[i-1], knots[i], partial_mean
        );
        if (curr_f_min > f_min) return argmin;
        argmin = curr_argmin;
        f_min = curr_f_min;
    }

    partial_mean -= 2 * knots[a_end-1] / K;
    ValueType curr_argmin;
    ValueType curr_f_min;
    std::tie(curr_argmin, curr_f_min) = quad_min(
        a_end, knots[a_end-1], a_right, partial_mean
    );
    return (curr_f_min > f_min) ? argmin : curr_argmin;
}

} // namespace optimization
} // namespace adelie_core