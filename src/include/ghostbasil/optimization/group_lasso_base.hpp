#pragma once
#include <ghostbasil/util/types.hpp>
#include <ghostbasil/util/macros.hpp>
#include <newton.hpp> // prob need to change location

namespace ghostbasil {
namespace group_lasso {

/**
 * Pack of buffers used for fit().
 * This class is purely for convenience purposes.
 */
template <class ValueType>
struct GroupLassoBufferPack 
{
    using value_t = ValueType;
    
    util::vec_type<value_t> buffer1;
    util::vec_type<value_t> buffer2;
    util::vec_type<value_t> buffer3;

    explicit GroupLassoBufferPack(
        size_t buffer_size
    )
        : GroupLassoBufferPack(
            buffer_size, buffer_size, buffer_size
        ) 
    {}

    explicit GroupLassoBufferPack(
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
 * Constructs active (feature) indices in increasing order
 * expanding group ranges as a dense vector.
 * The result is stored in out.
 * 
 * @param   pack    see GroupLassoCovParamPack.
 * @param   out     output vector.
 */
template <class PackType, class OutType>
GHOSTBASIL_STRONG_INLINE
void get_active_indices(
    const PackType& pack,
    OutType& out
)
{
    using index_t = typename PackType::index_t;
    using vec_t = util::vec_type<index_t>;
    
    const auto& active_set = *pack.active_set;
    const auto& active_order = *pack.active_order;
    const auto& strong_set = pack.strong_set;
    const auto& group_sizes = pack.group_sizes;
    const auto& groups = pack.groups;

    auto out_begin = out.data();
    for (size_t i = 0; i < active_order.size(); ++i) {
        const auto ss_idx = active_set[active_order[i]];
        const auto group = strong_set[ss_idx];
        const auto group_size = group_sizes[group];
        Eigen::Map<vec_t> seg(out_begin, group_size);
        seg = vec_t::LinSpaced(
            group_size, groups[group], groups[group] + group_size - 1
        );
        out_begin += group_size;
    }
    assert(out.size() == std::distance(out.data(), out_begin));
}

/**
 * Constructs active (feature) values in increasing index order.
 * The result is stored in out.
 * 
 * @param   pack    see GroupLassoCovParamPack.
 * @param   out     output vector.
 */
template <class PackType, class OutType>
GHOSTBASIL_STRONG_INLINE
void get_active_values(
    const PackType& pack,
    OutType& out 
)
{
    using value_t = typename PackType::value_t;
    using vec_t = util::vec_type<value_t>;

    const auto& active_set = *pack.active_set;
    const auto& active_order = *pack.active_order;
    const auto& strong_set = pack.strong_set;
    const auto& group_sizes = pack.group_sizes;
    const auto& strong_beta = pack.strong_beta;
    const auto& strong_begins = pack.strong_begins;

    auto out_begin = out.data();
    for (size_t i = 0; i < active_order.size(); ++i) {
        const auto ss_idx = active_set[active_order[i]];
        const auto group = strong_set[ss_idx];
        const auto group_size = group_sizes[group];
        Eigen::Map<vec_t> seg(out_begin, group_size);
        seg = strong_beta.segment(strong_begins[ss_idx], group_size);
        out_begin += group_size;
    }        
    assert(out.size() == std::distance(out.data(), out_begin));
}

/**
 * Computes the objective that we wish to minimize.
 * The objective is the quadratic loss + group-lasso regularization:
 * \f[
 *      \frac{1}{2} \sum_{ij} x_i^\top A_{ij} x_j - \sum_{i} x_i^\top r 
 *          + \lambda \sum_i p_i \left(
 *              \alpha ||x_i||_2 + \frac{1-\alpha}{2} ||x_i||_2^2
 *              \right)
 * \f]
 *          
 * @param   A       any square (p, p) matrix. 
 * @param   r       any vector (p,).
 * @param   groups  vector (G,) of indices marking the beginning of a group.
 * @param   group_sizes vector (G,) of group sizes.
 * @param   alpha       elastic net proportion.
 * @param   penalty penalty factor for each group.
 * @param   lmda    group-lasso regularization.
 * @param   beta    coefficient vector.
 */
template <class AType, class RType, 
          class GroupsType, class GroupSizesType,
          class ValueType, class PenaltyType, class BetaType>
GHOSTBASIL_STRONG_INLINE 
auto objective(
    const AType& A,
    const RType& r,
    const GroupsType& groups,
    const GroupSizesType& group_sizes,
    ValueType alpha,
    const PenaltyType& penalty,
    ValueType lmda,
    const BetaType& beta)
{
    ValueType p_ = 0.0;
    for (size_t j = 0; j < groups.size(); ++j) {
        const auto begin = groups[j];
        const auto size = group_sizes[j];
        const auto b_norm2 = beta.segment(begin, size).norm();
        p_ += penalty[j] * b_norm2 * (
            alpha + (1-alpha) / 2 * b_norm2
        );
    }
    p_ *= lmda;
    return 0.5 * A.quad_form(beta) - beta.dot(r) + p_;
}

/**
 * @brief Computes the objective given data.
 * 
 * @tparam XType            float matrix type.
 * @tparam YType            float vector type.
 * @tparam GroupsType       int vector type.
 * @tparam GroupSizesType   int vector type.
 * @tparam ValueType        float type.
 * @tparam PenaltyType      float vector type.
 * @tparam BetaType         float vector type.
 * @param X                 data matrix.
 * @param y                 response vector.
 * @param groups            group indices.
 * @param group_sizes       group sizes.
 * @param alpha             elastic net proportion.
 * @param penalty           penalty factor for each group. 
 * @param lmda              overall group-lasso penalty.
 * @param beta              coefficient vector.
 * @return objective value. 
 */
template <class XType, class YType, 
          class GroupsType, class GroupSizesType,
          class ValueType, class PenaltyType, class BetaType>
GHOSTBASIL_STRONG_INLINE 
auto objective_data(
    const XType& X,
    const YType& y,
    const GroupsType& groups,
    const GroupSizesType& group_sizes,
    ValueType alpha,
    const PenaltyType& penalty,
    ValueType lmda,
    const BetaType& beta)
{
    ValueType p_ = 0.0;
    for (size_t j = 0; j < groups.size(); ++j) {
        const auto begin = groups[j];
        const auto size = group_sizes[j];
        const auto b_norm2 = beta.segment(begin, size).norm();
        p_ += penalty[j] * b_norm2 * (
            alpha + (1-alpha) / 2 * b_norm2
        );
    }
    p_ *= lmda;
    Eigen::VectorXd resid = y - X * beta;
    return 0.5 * resid.squaredNorm() + p_;
}

/**
 * Updates the convergence measure using variance of each direction.
 * 
 * @param   convg_measure   convergence measure to update.
 * @param   del             vector difference in a group coefficient.
 * @param   var             vector of variance along each direction of coefficient.
 */
template <class ValueType, class DelType, class VarType>
GHOSTBASIL_STRONG_INLINE 
void update_convergence_measure(
    ValueType& convg_measure, 
    const DelType& del, 
    const VarType& var)
{
    const auto convg_measure_curr = del.dot(var.cwiseProduct(del)) / del.size();
    convg_measure = std::max(convg_measure, convg_measure_curr);
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
GHOSTBASIL_STRONG_INLINE
void update_rsq(
    ValueType& rsq,
    const DelType& del,
    const VarType& var,
    const RType& r
)
{
    rsq += (
        del.array() * (2 * r.array() - var.array() * del.array())
    ).sum();
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
GHOSTBASIL_STRONG_INLINE
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
    glstudy::newton_abs_solver(
        L, v, l1, l2, tol, max_iters,
        x, iters, buffer1, buffer2
    );
}

} // namespace group_lasso
} // namespace ghostbasil