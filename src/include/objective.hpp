#pragma once
#include <Eigen/Dense>

namespace glstudy {

/**
 * @brief Compute lower bound h_min >= 0 such that 
 * block_norm_objective(h_min) >= 0.
 * 
 * @tparam DiagType     vector type.
 * @tparam VType        vector type.
 * @tparam ValueType    float type.
 * @param vbuffer1      vector containing L + l2.
 * @param v             any vector of same length as vbuffer1.
 * @param l1            l1 regularization.
 * @return h_min 
 */
template <class DiagType, class VType, class ValueType>
inline
auto compute_h_min(
    const DiagType& vbuffer1,
    const VType& v,
    ValueType l1
)
{
    using value_t = ValueType;
    const value_t b = l1 * vbuffer1.sum();
    const value_t a = vbuffer1.squaredNorm();
    const value_t v_l1 = v.template lpNorm<1>();
    const value_t c = l1 * l1 * vbuffer1.size() - v_l1 * v_l1;
    const value_t discr = b*b - a*c;
    value_t h_min = (discr > -1e-12) ? 
        (-b + std::sqrt(std::max(discr, 0.0))) / a : 0.0;
    
    // Otherwise, if h <= 0, we know at least 0 is a reasonable solution.
    // The only case h <= 0 is when 0 is already close to the solution.
    h_min = std::max(h_min, 0.0);
    return h_min;
}

/**
 * @brief 
 * Compute upper bound h_max >= 0 such that block_norm_objective(h_max) <= 0.
 * NOTE: if zero_tol > 0,
 * the result may NOT be a true upper bound in the sense that objective(result) <= 0.
 * 
 * @tparam DiagType     vector type.
 * @tparam VType        vector type.
 * @tparam ValueType    float type.
 * @param vbuffer1      vector containing L + l2.
 * @param v             any vector of same length as vbuffer1.
 * @param l1            l1 regularization.
 * @param zero_tol      if a float is <= zero_tol, it is considered to be 0.
 * @return (h_max, vbuffer1_min_nzn)
 *  h_max: the upper bound
 *  vbuffer1_min_nzn:   smallest value in vbuffer1 among non-zero values based on zero_tol.
 */
template <class DiagType, class VType, class ValueType>
inline
auto compute_h_max(
    const DiagType& vbuffer1,
    const VType& v,
    ValueType l1,
    ValueType zero_tol=1e-10
)
{
    using value_t = ValueType;

    const value_t vbuffer1_min = vbuffer1.minCoeff();

    value_t vbuffer1_min_nzn = std::numeric_limits<value_t>::infinity();
    value_t h_max = 0;

    // If L+l2 have entries <= threshold,
    // find h_max with more numerically-stable routine.
    // If threshold > 0, there is NO guarantee that f(h_max) <= 0, 
    // but we will use this to bisect and find an h where f(h) >= 0,
    // so we don't necessarily need h_max to be f(h_max) <= 0.
    if (vbuffer1_min <= zero_tol) {
        value_t denom = 0;
        for (int i = 0; i < vbuffer1.size(); ++i) {
            const bool is_nonzero = vbuffer1[i] > zero_tol;
            const auto vi2 = v[i] * v[i];
            h_max += is_nonzero ? vi2 / (vbuffer1[i] * vbuffer1[i]) : 0;
            denom += is_nonzero ? 0 : vi2; 
            vbuffer1_min_nzn = is_nonzero ? std::min(vbuffer1_min_nzn, vbuffer1[i]) : vbuffer1_min_nzn;
        }
        h_max = std::sqrt(std::abs(h_max / (1.0 - denom / (l1 * l1))));
    } else {
        vbuffer1_min_nzn = vbuffer1_min;
        h_max = (v.array() / vbuffer1.array()).matrix().norm();
    }

    return std::make_pair(h_max, vbuffer1_min_nzn);
}
    
/*
 * Block coordinate descent norm objective
 */
template <class ValueType, class DiagType, class VType>
inline
auto block_norm_objective(
    ValueType h,
    const DiagType& D,
    const VType& v,
    ValueType l1
)
{
    return (v.array() / (D.array() * h + l1)).matrix().squaredNorm() - 1;
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