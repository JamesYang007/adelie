#pragma once
#include <Eigen/SVD>
#include <adelie_core/util/types.hpp>
#include <adelie_core/util/macros.hpp>

namespace adelie_core {
namespace solver {

/**
 * TODO: not currently used
 */
template <class BetasType, class RotationsType, class IndexType>
ADELIE_CORE_STRONG_INLINE
void untransform_beta(
    BetasType& betas,
    RotationsType& rotations,
    const Eigen::Ref<const util::rowvec_type<IndexType>>& group_sizes,
    const Eigen::Ref<const util::rowvec_type<IndexType>>& screen_set,
    const Eigen::Ref<const util::rowvec_type<IndexType>>& active_set,
    const Eigen::Ref<const util::rowvec_type<IndexType>>& active_order
)
{
    for (auto& beta : betas) {
        using value_t = typename std::decay_t<decltype(beta)>::Scalar;
        Eigen::Map<util::rowvec_type<value_t>> beta_map(
            beta.innerNonZeroPtr(), 
            beta.nonZeros()
        );
        int curr_idx = 0;
        for (const auto i : active_order) {
            const auto ss_idx = active_set[i];
            const auto gs = group_sizes[screen_set[ss_idx]];
            auto beta_i = beta_map.segment(curr_idx, gs); 
            beta_i.matrix() = beta_i.matrix() * rotations[ss_idx].transpose();
            curr_idx += gs;
        }
    }
}

} // namespace solver
} // namespace adelie_core