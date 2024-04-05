#pragma once
#include <adelie_core/util/types.hpp>

namespace adelie_core {
namespace optimization {

/**
 * Solves NNLS using full quadratic component.
 * 
 * @param   iters   number of iterations.
 * @param   x       initial value and also the output.
 * @param   grad    gradient v - Q * x.
 */
template <class QuadType, class ValueType, class OutType>
void nnls_cov_full(
    const QuadType& quad,
    ValueType l2,
    size_t max_iters,
    ValueType tol,
    size_t& iters,
    OutType& x,
    OutType& grad
)
{
    using value_t = ValueType;

    const auto d = x.size();

    iters = 0;

    while (iters < max_iters) {
        value_t convg_measure = 0;
        ++iters;
        for (int i = 0; i < d; ++i) {
            if (quad(i,i) + l2 <= 0) continue;
            const auto xi = x[i];
            const auto qii = quad(i,i) + l2;
            const auto gi = grad[i];
            const auto xi_new = std::max<value_t>(xi + gi / qii, 0);
            const auto del = xi_new - xi;
            convg_measure = std::max<value_t>(convg_measure, qii * del * del);
            x[i] = xi_new;
            if constexpr (std::decay_t<QuadType>::IsRowMajor) {
                grad -= del * (quad.array().row(i) + l2);
            } else {
                grad -= del * (quad.array().col(i) + l2);
            }
        }
        if (convg_measure < tol) break;
    }
}

} // namespace optimization
} // namespace adelie_core