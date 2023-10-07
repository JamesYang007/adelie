#pragma once
#include <adelie_core/util/types.hpp>
#include <adelie_core/util/macros.hpp>

namespace adelie_core {
namespace grpnet {

template <class ValueType, class AbsGradType, class PenaltyType>
ADELIE_CORE_STRONG_INLINE
auto lambda_max(
    const AbsGradType& abs_grad,
    ValueType alpha,
    const PenaltyType& penalty
)
{
    using value_t = ValueType;
    using vec_value_t = util::rowvec_type<value_t>;
    const auto factor = (alpha <= 0) ? 1e-3 : alpha;
    return vec_value_t::NullaryExpr(
        abs_grad.size(), [&](auto i) {
            return (penalty[i] <= 0.0) ? 0.0 : abs_grad[i] / penalty[i];
        }
    ).maxCoeff() / factor;
}

template <class ValueType, class OutType>
ADELIE_CORE_STRONG_INLINE
void create_lambdas(
    size_t max_n_lambdas,
    ValueType min_ratio,
    ValueType lmda_max,
    OutType& out
)
{
    using value_t = ValueType;
    using vec_value_t = util::rowvec_type<value_t>;

    // lmda_seq = [l_max, l_max * f, l_max * f^2, ..., l_max * f^(max_n_lambdas-1)]
    // l_max is the smallest lambda such that the penalized features (penalty > 0)
    // have 0 coefficients (assuming alpha > 0). The logic is still coherent when alpha = 0.
    auto log_factor = std::log(min_ratio) * static_cast<value_t>(1.0)/(max_n_lambdas-1);
    out = lmda_max * (
        log_factor * vec_value_t::LinSpaced(max_n_lambdas, 0, max_n_lambdas-1)
    ).exp();
}

} // namespace grpnet
} // namespace adelie_core