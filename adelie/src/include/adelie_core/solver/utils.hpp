#pragma once
#include <adelie_core/util/types.hpp>

namespace adelie_core {
namespace solver {

template <class LmdaPathType, class ValueType>
void generate_lmda_path(
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