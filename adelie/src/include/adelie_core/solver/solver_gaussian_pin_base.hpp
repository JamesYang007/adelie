#pragma once
#include <adelie_core/configs.hpp>
#include <adelie_core/bcd/unconstrained/newton.hpp>
#include <adelie_core/util/macros.hpp>
#include <adelie_core/util/types.hpp>

namespace adelie_core {
namespace solver {
namespace gaussian {
namespace pin {

/**
 * Pack of buffers used in solvers.
 * This class is purely for convenience purposes.
 */
template <class ValueType, class IndexType>
struct GaussianPinBufferPack 
{
    using value_t = ValueType;
    using index_t = IndexType;
    
    util::rowvec_type<value_t> buffer1;
    util::rowvec_type<value_t> buffer2;
    util::rowvec_type<value_t> buffer3;
    util::rowvec_type<value_t> buffer4;
    util::rowvec_type<uint64_t> constraint_buffer;

    std::vector<index_t> active_beta_indices;
    std::vector<value_t> active_beta_ordered;

    explicit GaussianPinBufferPack(
        size_t buffer1_size, 
        size_t buffer2_size,
        size_t buffer3_size,
        size_t buffer4_size,
        size_t constraint_buffer_size,
        size_t active_beta_size
    ): 
        buffer1(buffer1_size),
        buffer2(buffer2_size),
        buffer3(buffer3_size),
        buffer4(buffer4_size),
        constraint_buffer(constraint_buffer_size)
    {
        // allocate buffers for optimization
        active_beta_indices.reserve(active_beta_size);
        active_beta_ordered.reserve(active_beta_size);
    }
};

/**
 * Constructs a sparse vector containing all active values.
 * 
 * @param   state    see StateGaussianPinNaive.
 * @param   indices     increasing order of indices with active values.
 * @param   values      corresponding active values to indices.
 */
template <class StateType, class VecIndexType, class VecValueType>
inline void sparsify_active_beta(
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
    const auto& constraints = *state.constraints;
    const auto& groups = state.groups;
    const auto& group_sizes = state.group_sizes;
    const auto& screen_beta = state.screen_beta;
    const auto& screen_begins = state.screen_begins;

    auto idxs_begin = indices.data();
    auto vals_begin = values.data();
    for (size_t i = 0; i < active_order.size(); ++i) {
        const auto ss_idx = active_set[active_order[i]];
        const auto group = screen_set[ss_idx];
        const auto group_size = group_sizes[group];
        const auto constraint = constraints[group];
        Eigen::Map<vec_index_t> idxs_seg(idxs_begin, group_size);
        Eigen::Map<vec_value_t> vals_seg(vals_begin, group_size);
        idxs_seg = vec_index_t::LinSpaced(
            group_size, groups[group], groups[group] + group_size - 1
        );
        vals_seg = screen_beta.segment(screen_begins[ss_idx], group_size);
        if (Configs::project && constraint) constraint->project(vals_seg);
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

template <
    class LType, 
    class VType, 
    class ValueType, 
    class XType, 
    class BufferType
>
ADELIE_CORE_STRONG_INLINE
void update_coordinate(
    XType& x,
    const LType& L,
    const VType& v,
    ValueType l1,
    ValueType l2,
    ValueType tol,
    size_t max_iters,
    BufferType& buffer1,
    BufferType& buffer2
)
{
    size_t iters;
    bcd::unconstrained::newton_solver(
        L, v, l1, l2, tol, max_iters,
        x, iters, buffer1, buffer2
    );
    if (iters >= max_iters) {
        throw util::adelie_core_solver_error(
            "Newton-ABS max iterations reached! "
            "Try increasing newton_max_iters."
        );
    }
}

template <class ValueType>
ADELIE_CORE_STRONG_INLINE
void update_coordinate(
    ValueType& coeff,
    ValueType x_var,
    ValueType grad,
    ValueType l1,
    ValueType l2
)
{
    const auto denom = x_var + l2;
    const auto u = grad;
    const auto v = std::abs(u) - l1;
    coeff = (v > 0.0) ? std::copysign(v,u)/denom : 0;
}

} // namespace pin
} // namespace gaussian
} // namespace solver
} // namespace adelie_core