#pragma once
#include <cmath>
#include <adelie_core/util/types.hpp>

namespace adelie_core {
namespace bcd {

/**
 * @brief Compute lower bound h_min >= 0 such that 
 * root_function(h_min) >= 0.
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
auto root_lower_bound(
    const DiagType& vbuffer1,
    const VType& v,
    ValueType l1
)
{
    using value_t = ValueType;
    const value_t b = l1 * vbuffer1.sum();
    const value_t a = vbuffer1.matrix().squaredNorm();
    const value_t v_l1 = v.matrix().template lpNorm<1>();
    const value_t c = l1 * l1 * vbuffer1.size() - v_l1 * v_l1;
    const value_t discr = b*b - a*c;
    value_t h_min = (discr > -1e-12) ? 
        (-b + std::sqrt(std::max<value_t>(discr, 0.0))) / a : 0.0;
    
    // Otherwise, if h <= 0, we know at least 0 is a reasonable solution.
    // The only case h <= 0 is when 0 is already close to the solution.
    h_min = std::max<value_t>(h_min, 0.0);
    return h_min;
}

/**
 * @brief 
 * Compute upper bound h_max >= 0 such that root_function(h_max) <= 0.
 * NOTE: if zero_tol > 0,
 * the result may NOT be a true upper bound in the sense that group_elnet_objective(result) <= 0.
 * 
 * @tparam DiagType     vector type.
 * @tparam VType        vector type.
 * @tparam ValueType    float type.
 * @param vbuffer1      vector containing L + l2.
 * @param v             any vector of same length as vbuffer1.
 * @param zero_tol      if a float is <= zero_tol, it is considered to be 0.
 * @return (h_max, vbuffer1_min_nnz)
 *  h_max: the upper bound
 *  vbuffer1_min_nnz:   smallest value in vbuffer1 among non-zero values based on zero_tol.
 */
template <class DiagType, class VType, class ValueType>
inline
auto root_upper_bound(
    const DiagType& vbuffer1,
    const VType& v,
    ValueType l1,
    ValueType zero_tol=1e-14
)
{
    using value_t = ValueType;

    const value_t vbuffer1_min = vbuffer1.minCoeff();

    value_t vbuffer1_min_nnz = std::numeric_limits<value_t>::infinity();
    value_t h_max = 0;
    value_t v_S = 0;

    // If L+l2 have entries <= threshold,
    // find h_max with more numerically-stable routine.
    // If threshold > 0, there is NO guarantee that f(h_max) <= 0, 
    // but we will use this to bisect and find an h where f(h) >= 0,
    // so we don't necessarily need h_max to be f(h_max) <= 0.
    if (vbuffer1_min <= zero_tol) {
        for (int i = 0; i < vbuffer1.size(); ++i) {
            const bool is_nonzero = vbuffer1[i] > zero_tol;
            const auto vi2 = v[i] * v[i];
            h_max += is_nonzero ? (vi2 / (vbuffer1[i] * vbuffer1[i])) : 0;
            v_S += (vbuffer1[i] <= 0) ? vi2 : 0;
            vbuffer1_min_nnz = is_nonzero ? std::min(vbuffer1_min_nnz, vbuffer1[i]) : vbuffer1_min_nnz;
        }
        h_max = std::sqrt(std::max<value_t>(h_max / (1 - v_S / (l1 * l1)), 0));
    } else {
        vbuffer1_min_nnz = vbuffer1_min;
        h_max = (v / vbuffer1).matrix().norm();
    }

    return std::make_pair(h_max, vbuffer1_min_nnz);
}

template <class ValueType, class DiagType, class VType>
inline
auto root_function(
    ValueType h,
    const DiagType& D,
    const VType& v,
    ValueType l1
)
{
    return (v.array() / (D.array() * h + l1)).matrix().squaredNorm() - 1;
}

} // namespace bcd
} // namespace adelie_core