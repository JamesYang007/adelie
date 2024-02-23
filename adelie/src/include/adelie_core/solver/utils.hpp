#pragma once
#include <adelie_core/util/types.hpp>

namespace adelie_core {
namespace solver {

template <class AbsGradType, class ValueType, class PenaltyType>
auto compute_lmda_max(
    const AbsGradType& abs_grad,
    ValueType alpha,
    const PenaltyType& penalty,
    ValueType ridge_scale = 1e-3
)
{
    using vec_value_t = std::decay_t<AbsGradType>;
    const auto factor = (alpha <= 0) ? ridge_scale : alpha;
    return vec_value_t::NullaryExpr(
        abs_grad.size(),
        [&](auto i) { 
            return (penalty[i] <= 0.0) ? 0.0 : abs_grad[i] / penalty[i];
        }
    ).maxCoeff() / factor;
}

template <class LmdaPathType, class ValueType>
void compute_lmda_path(
    LmdaPathType& lmda_path,
    ValueType min_ratio,
    ValueType lmda_max
)
{
    using vec_value_t = util::rowvec_type<ValueType>;

    const auto lmda_path_size = lmda_path.size();
    if (lmda_path_size > 1) {
        const auto log_factor = std::log(min_ratio) / (lmda_path_size - 1);
        lmda_path = lmda_max * (log_factor * vec_value_t::LinSpaced(
            lmda_path_size, 0, lmda_path_size-1
        )).exp();
    }
    lmda_path[0] = lmda_max; // for numerical stability
}

} // namespace solver
} // namespace adelie_core