#pragma once
#include <Eigen/Dense>

namespace glstudy {

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
 * @param   groups  see description in GroupLassoParamPack.
 * @param   group_sizes see description in GroupLassoParamPack.
 * @param   alpha       elastic net proportion.
 * @param   penalty penalty factor for each group.
 * @param   lmda    group-lasso regularization.
 * @param   beta    coefficient vector.
 */
template <class AType, class RType, 
          class GroupsType, class GroupSizesType,
          class ValueType, class PenaltyType, class BetaType>
inline 
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
    for (size_t j = 0; j < groups.size()-1; ++j) {
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

} // namespace glstudy