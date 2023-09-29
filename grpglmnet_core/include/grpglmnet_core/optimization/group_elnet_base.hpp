#pragma once
#include <grpglmnet_core/util/types.hpp>
#include <grpglmnet_core/util/macros.hpp>
#include <grpglmnet_core/optimization/newton.hpp>

namespace grpglmnet_core {

/**
 * Pack of buffers used for fit().
 * This class is purely for convenience purposes.
 */
template <class ValueType>
struct GroupElnetBufferPack 
{
    using value_t = ValueType;
    
    util::rowvec_type<value_t> buffer1;
    util::rowvec_type<value_t> buffer2;
    util::rowvec_type<value_t> buffer3;

    explicit GroupElnetBufferPack(
        size_t buffer_size
    )
        : GroupElnetBufferPack(
            buffer_size, buffer_size, buffer_size
        ) 
    {}

    explicit GroupElnetBufferPack(
            size_t buffer1_size, 
            size_t buffer2_size,
            size_t buffer3_size
    )
        : buffer1(buffer1_size),
          buffer2(buffer2_size),
          buffer3(buffer3_size)
    {}
};

/**
 * Constructs a sparse vector containing all active values.
 * 
 * @param   pack    see GroupElnetState.
 * @param   indices     increasing order of indices with active values.
 * @param   values      corresponding active values to indices.
 */
template <class PackType, class VecIndexType, class VecValueType>
GRPGLMNET_CORE_STRONG_INLINE
void sparsify_active_beta(
    const PackType& pack,
    VecIndexType& indices,
    VecValueType& values
)
{
    using index_t = typename PackType::index_t;
    using value_t = typename PackType::value_t;
    using vec_index_t = util::rowvec_type<index_t>;
    using vec_value_t = util::rowvec_type<value_t>;

    const auto& active_set = pack.active_set;
    const auto& active_order = pack.active_order;
    const auto& strong_set = pack.strong_set;
    const auto& group_sizes = pack.group_sizes;
    const auto& groups = pack.groups;
    const auto& strong_beta = pack.strong_beta;
    const auto& strong_begins = pack.strong_begins;

    indices.resize(active_order.size());
    values.resize(active_order.size());

    auto idxs_begin = indices.data();
    auto vals_begin = values.data();
    for (size_t i = 0; i < active_order.size(); ++i) {
        const auto ss_idx = active_set[active_order[i]];
        const auto group = strong_set[ss_idx];
        const auto group_size = group_sizes[group];
        Eigen::Map<vec_index_t> idxs_seg(idxs_begin, group_size);
        Eigen::Map<vec_value_t> vals_seg(vals_begin, group_size);
        idxs_seg = vec_index_t::LinSpaced(
            group_size, groups[group], groups[group] + group_size - 1
        );
        vals_seg = strong_beta.segment(strong_begins[ss_idx], group_size);
        idxs_begin += group_size;
        vals_begin += group_size;
    }        
    assert(indices.size() == std::distance(indices.data(), idxs_begin));
    assert(values.size() == std::distance(values.data(), vals_begin));
}

/**
 * Updates the convergence measure using variance of each direction.
 * 
 * @param   convg_measure   convergence measure to update.
 * @param   del             vector difference in a group coefficient.
 * @param   var             vector of variance along each direction of coefficient.
 */
template <class ValueType, class DelType, class VarType>
GRPGLMNET_CORE_STRONG_INLINE 
void update_convergence_measure(
    ValueType& convg_measure, 
    const DelType& del, 
    const VarType& var)
{
    const auto convg_measure_curr = (var * del.square()).sum() / del.size();
    convg_measure = std::max(convg_measure, convg_measure_curr);
}

/**
 * Updates the convergence measure using variance of feature and coefficient change.
 * NOTE: this is for lasso specifically.
 *
 * @param   convg_measure   current convergence measure to update.
 * @param   coeff_diff      new coefficient minus old coefficient.
 * @param   x_var           variance of feature. A[k,k] where k is the feature corresponding to coeff_diff.
 */
template <class ValueType>
GRPGLMNET_CORE_STRONG_INLINE
void update_convergence_measure(
    ValueType& convg_measure,
    ValueType coeff_diff,
    ValueType x_var
)
{
    const auto convg_measure_curr = x_var * coeff_diff * coeff_diff;
    convg_measure = std::max(convg_measure_curr, convg_measure);
}

/**
 * Updates \f$R^2\f$ given the group variance vector, 
 * group coefficient difference (new minus old), 
 * and the current residual vector.
 * 
 * @param   rsq     \f$R^2\f$ to update.
 * @param   del     new coefficient minus old coefficient.
 * @param   var     variance along each coordinate of group.
 * @param   r       current residual correlation vector for group.
 */
template <class ValueType, class DelType, 
          class VarType, class RType>
GRPGLMNET_CORE_STRONG_INLINE
void update_rsq(
    ValueType& rsq,
    const DelType& del,
    const VarType& var,
    const RType& r
)
{
    rsq += (del * (2 * r - var * del)).sum();
}

/**
 * Increments rsq with the difference in R^2.
 * NOTE: this is for lasso specifically.
 *
 * @param   rsq         R^2 to update.
 * @param   old_coeff   old coefficient.
 * @param   new_coeff   new coefficient.
 * @param   x_var       variance of feature (A[k,k]).
 * @param   grad        (negative) gradient corresponding to the coefficient.
 * @param   s           regularization of A towards identity.
 */
template <class ValueType>
GRPGLMNET_CORE_STRONG_INLINE
void update_rsq(
        ValueType& rsq, 
        ValueType old_coeff, 
        ValueType new_coeff, 
        ValueType x_var, 
        ValueType grad)
{
    const auto del = new_coeff - old_coeff;
    rsq += del * (2 * grad - del * x_var);
}


/**
 * Solves the solution for the equation (w.r.t. \f$x\f$):
 * \f[
 *      minimize \frac{1}{2} x^\top L x - x^\top v 
 *          + l_1 ||x||_2 + \frac{l_2}{2} ||x||_2^2
 * \f]
 *      
 * @param   L       vector representing a diagonal PSD matrix.
 *                  Must have max(L + s) > 0. 
 *                  L.size() <= buffer1.size().
 * @param   v       any vector.  
 * @param   l1      L2-norm penalty. Must be >= 0.
 * @param   l2      L2 penalty. Must be >= 0.
 * @param   tol         Newton's method tolerance of closeness to 0.
 * @param   max_iters   maximum number of iterations of Newton's method.
 * @param   x           solution vector.
 * @param   iters       number of Newton's method iterations taken.
 * @param   buffer1     any vector with L.size() <= buffer1.size().
 * @param   buffer2     any vector with L.size() <= buffer2.size().
 */
template <class LType, class VType, class ValueType, 
          class XType, class BufferType>
GRPGLMNET_CORE_STRONG_INLINE
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
    grpglmnet_core::newton_abs_solver(
        L, v, l1, l2, tol, max_iters,
        x, iters, buffer1, buffer2
    );
}

/**
 * Updates the coefficient given the current state via coordinate descent rule.
 * NOTE: this is for lasso specifically.
 *
 * @param   coeff   current coefficient to update.
 * @param   x_var   variance of feature. A[k,k] where k is the feature corresponding to coeff.
 * @param   l1      L1 regularization part in elastic net.
 * @param   l2      L2 regularization part in elastic net.
 * @param   penalty penalty value for current coefficient.
 * @param   grad    current (negative) gradient for coeff.
 */
template <class ValueType>
GRPGLMNET_CORE_STRONG_INLINE
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
 *      delta_u < cond_0_thresh 
 *      AND
 *      (delta_u - delta_m) < cond_1_thresh
 *
 * @param   rsq_l   third to last R^2 value.
 * @param   rsq_m   second to last R^2 value.
 * @param   rsq_u   last R^2 value.
 * @param   cond_0_thresh   threshold for derivative condition.
 * @param   cond_1_thresh   threshold for second derivative condition.
 */
template <class ValueType>
GRPGLMNET_CORE_STRONG_INLINE
bool check_early_stop_rsq(
    ValueType rsq_l,
    ValueType rsq_m,
    ValueType rsq_u,
    ValueType cond_0_thresh = 1e-5,
    ValueType cond_1_thresh = 1e-5
)
{
    const auto delta_u = (rsq_u-rsq_m);
    const auto delta_m = (rsq_m-rsq_l);
    return ((delta_u <= cond_0_thresh*rsq_u) &&
            ((delta_m*rsq_u-delta_u*rsq_m) <= cond_1_thresh*rsq_m*rsq_u));
}

template <class XType, class YType, 
          class GroupsType, class GroupSizesType,
          class ValueType, class PenaltyType, class BetaType>
inline
auto group_elnet_objective(
    const BetaType& beta,
    const XType& X,
    const YType& y,
    const GroupsType& groups,
    const GroupSizesType& group_sizes,
    ValueType lmda,
    ValueType alpha,
    const PenaltyType& penalty
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
    return 0.5 * (y.matrix() - X * beta.matrix()).squaredNorm() + p_;
}

} // namespace grpglmnet_core