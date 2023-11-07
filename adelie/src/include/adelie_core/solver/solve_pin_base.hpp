#pragma once
#include <adelie_core/util/types.hpp>
#include <adelie_core/util/macros.hpp>
#include <adelie_core/bcd.hpp>

namespace adelie_core {
namespace solver {

/**
 * Pack of buffers used in solvers.
 * This class is purely for convenience purposes.
 */
template <class ValueType>
struct SolvePinBufferPack 
{
    using value_t = ValueType;
    
    util::rowvec_type<value_t> buffer1;
    util::rowvec_type<value_t> buffer2;
    util::rowvec_type<value_t> buffer3;
    util::rowvec_type<value_t> buffer4;

    explicit SolvePinBufferPack(
        size_t buffer_size,
        size_t n
    ): 
        SolvePinBufferPack(
            buffer_size, buffer_size, buffer_size, n
        ) 
    {}

    explicit SolvePinBufferPack(
            size_t buffer1_size, 
            size_t buffer2_size,
            size_t buffer3_size,
            size_t buffer4_size
    ): 
        buffer1(buffer1_size),
        buffer2(buffer2_size),
        buffer3(buffer3_size),
        buffer4(buffer4_size)
    {}
};

/**
 * Constructs a sparse vector containing all active values.
 * 
 * @param   state    see StatePinNaive.
 * @param   indices     increasing order of indices with active values.
 * @param   values      corresponding active values to indices.
 */
template <class StateType, class VecIndexType, class VecValueType>
ADELIE_CORE_STRONG_INLINE
void sparsify_active_beta(
    const StateType& state,
    VecIndexType& indices,
    VecValueType& values
)
{
    using index_t = typename StateType::index_t;
    using value_t = typename StateType::value_t;
    using vec_index_t = util::rowvec_type<index_t>;
    using vec_value_t = util::rowvec_type<value_t>;

    const auto& active_set = state.active_set;
    const auto& active_order = state.active_order;
    const auto& screen_set = state.screen_set;
    const auto& group_sizes = state.group_sizes;
    const auto& groups = state.groups;
    const auto& screen_beta = state.screen_beta;
    const auto& screen_begins = state.screen_begins;

    auto idxs_begin = indices.data();
    auto vals_begin = values.data();
    for (size_t i = 0; i < active_order.size(); ++i) {
        const auto ss_idx = active_set[active_order[i]];
        const auto group = screen_set[ss_idx];
        const auto group_size = group_sizes[group];
        Eigen::Map<vec_index_t> idxs_seg(idxs_begin, group_size);
        Eigen::Map<vec_value_t> vals_seg(vals_begin, group_size);
        idxs_seg = vec_index_t::LinSpaced(
            group_size, groups[group], groups[group] + group_size - 1
        );
        vals_seg = screen_beta.segment(screen_begins[ss_idx], group_size);
        idxs_begin += group_size;
        vals_begin += group_size;
    }        
    assert(indices.size() == std::distance(indices.data(), idxs_begin));
    assert(values.size() == std::distance(values.data(), vals_begin));
}

template <class ValueType, class DelType, class VarType>
ADELIE_CORE_STRONG_INLINE 
void update_convergence_measure(
    ValueType& convg_measure, 
    const DelType& del, 
    const VarType& var
)
{
    const auto convg_measure_curr = (var * del.square()).sum() / del.size();
    convg_measure = std::max(convg_measure, convg_measure_curr);
}

template <class ValueType>
ADELIE_CORE_STRONG_INLINE
void update_convergence_measure(
    ValueType& convg_measure,
    ValueType coeff_diff,
    ValueType x_var
)
{
    const auto convg_measure_curr = x_var * coeff_diff * coeff_diff;
    convg_measure = std::max(convg_measure_curr, convg_measure);
}

template <class ValueType, class DelType, class XVarType, class GradType>
ADELIE_CORE_STRONG_INLINE
void update_rsq(
    ValueType& rsq,
    const DelType& del,
    const XVarType& x_var,
    const GradType& grad
)
{
    rsq += (del * (2 * grad - del * x_var)).sum();
}

template <class ValueType>
ADELIE_CORE_STRONG_INLINE
void update_rsq(
    ValueType& rsq, 
    ValueType del,
    ValueType x_var, 
    ValueType grad
)
{
    rsq += del * (2 * grad - del * x_var);
}

template <class LType, class VType, class ValueType, 
          class XType, class BufferType>
ADELIE_CORE_STRONG_INLINE
void update_coefficients(
    const LType& L,
    const VType& v,
    ValueType l1,
    ValueType l2,
    ValueType tol,
    size_t max_iters,
    XType& x,
    size_t& iters,
    BufferType& buffer1,
    BufferType& buffer2
)
{
    bcd::newton_abs_solver(
        L, v, l1, l2, tol, max_iters,
        x, iters, buffer1, buffer2
    );
}

template <class ValueType>
ADELIE_CORE_STRONG_INLINE
void update_coefficient(
    ValueType& coeff,
    ValueType x_var,
    ValueType l1,
    ValueType l2,
    ValueType penalty,
    ValueType grad
)
{
    const auto denom = x_var + l2 * penalty;
    const auto u = grad + coeff * x_var;
    const auto v = std::abs(u) - l1 * penalty;
    coeff = (v > 0.0) ? std::copysign(v,u)/denom : 0;
}

/**
 * Checks early stopping based on R^2 values.
 * Returns true (early stopping should occur) if both are true:
 *
 *      delta_u := (R^2_u - R^2_m)/R^2_u
 *      delta_m := (R^2_m - R^2_l)/R^2_m 
 *      delta_u < rsq_slope_tol 
 *      AND
 *      (delta_u - delta_m) < rsq_curv_tol
 *
 * @param   rsq_l   third to last R^2 value.
 * @param   rsq_m   second to last R^2 value.
 * @param   rsq_u   last R^2 value.
 * @param   rsq_slope_tol   threshold for derivative condition.
 * @param   rsq_curv_tol   threshold for second derivative condition.
 */
template <class ValueType>
ADELIE_CORE_STRONG_INLINE
bool check_early_stop_rsq(
    ValueType rsq_l,
    ValueType rsq_m,
    ValueType rsq_u,
    ValueType rsq_slope_tol = 1e-5,
    ValueType rsq_curv_tol = 1e-5
)
{
    const auto delta_u = (rsq_u-rsq_m);
    const auto delta_m = (rsq_m-rsq_l);
    return ((delta_u <= rsq_slope_tol*rsq_u) &&
            ((delta_m*rsq_u-delta_u*rsq_m) <= rsq_curv_tol*rsq_m*rsq_u));
}

template <class ValueType, class IndexType> 
inline
auto objective(
    ValueType beta0, 
    const Eigen::Ref<const util::rowvec_type<ValueType>>& beta,
    const Eigen::Ref<const util::rowmat_type<ValueType>>& X,
    const Eigen::Ref<const util::rowvec_type<ValueType>>& y,
    const Eigen::Ref<const util::rowvec_type<IndexType>>& groups,
    const Eigen::Ref<const util::rowvec_type<IndexType>>& group_sizes,
    ValueType lmda,
    ValueType alpha,
    const Eigen::Ref<const util::rowvec_type<ValueType>>& penalty,
    const Eigen::Ref<const util::rowvec_type<ValueType>>& weights
)
{
    ValueType p_ = 0.0;
    for (int j = 0; j < groups.size(); ++j) {
        const auto begin = groups[j];
        const auto size = group_sizes[j];
        const auto b_norm2 = beta.segment(begin, size).matrix().norm();
        p_ += penalty[j] * b_norm2 * (
            alpha + 0.5 * (1-alpha) * b_norm2
        );
    }
    p_ *= lmda;
    util::rowvec_type<ValueType> resid = (y.matrix() - beta.matrix() * X.transpose()).array() - beta0;
    return 0.5 * (weights * resid.square()).sum() + p_;
}

} // namespace solver
} // namespace adelie_core