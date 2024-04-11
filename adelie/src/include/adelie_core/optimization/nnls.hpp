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
template <class QuadType, class ValueType, class XType, 
          class GradType, class EarlyExitType>
void nnls_cov_full(
    const QuadType& quad,
    size_t max_iters,
    ValueType tol,
    ValueType dtol,
    size_t& iters,
    XType& x,
    GradType& grad,
    ValueType& loss,
    EarlyExitType early_exit_f
)
{
    using value_t = ValueType;

    const auto d = x.size();

    iters = 0;

    while (iters < max_iters) {
        value_t convg_measure = 0;
        ++iters;
        for (int i = 0; i < d; ++i) {
            if (early_exit_f()) continue;
            const auto qii = quad(i,i);
            if (qii <= 0) { 
                x[i] = std::max<value_t>(x[i], 0); 
                continue;
            }
            const auto xi = x[i];
            const auto gi = grad[i];
            const auto xi_new = std::max<value_t>(xi + gi / qii, 0);
            const auto del = xi_new - xi;
            if (std::abs(del) <= dtol) continue;
            const auto scaled_del_sq = qii * del * del; 
            convg_measure = std::max<value_t>(convg_measure, scaled_del_sq);
            loss -= del * gi - 0.5 * scaled_del_sq;
            x[i] = xi_new;
            if constexpr (std::decay_t<QuadType>::IsRowMajor) {
                grad -= del * quad.array().row(i);
            } else {
                grad -= del * quad.array().col(i);
            }
        }
        if (convg_measure < tol) break;
    }
}

/**
 * Solves NNLS in regression form.
 * 
 * @param   iters   number of iterations.
 * @param   x       initial value and also the output.
 * @param   grad    gradient v - Q * x.
 */
template <class XType, class XVarsType, 
          class ValueType, class BetaType,
          class ResidType, class EarlyExitType, class SkipType>
void nnls_naive(
    const XType& X,
    const XVarsType& X_vars,
    size_t max_iters,
    ValueType tol,
    ValueType dtol,
    size_t& iters,
    BetaType& beta,
    ResidType& resid,
    ValueType& loss,
    EarlyExitType early_exit_f,
    SkipType skip_f
)
{
    using value_t = ValueType;

    const auto d = beta.size();

    iters = 0;

    while (iters < max_iters) {
        value_t convg_measure = 0;
        ++iters;
        for (int i = 0; i < d; ++i) {
            if (early_exit_f()) return;
            if (skip_f(i)) continue;
            const auto X_vars_i = X_vars[i];
            if (X_vars_i <= 0) { 
                beta[i] = std::max<value_t>(beta[i], 0); 
                continue;
            }
            const auto bi = beta[i];
            const auto gi = X.col(i).dot(resid.matrix());
            const auto bi_new = std::max<value_t>(bi + gi / X_vars_i, 0);
            const auto del = bi_new - bi;
            if (std::abs(del) <= dtol) continue;
            const auto scaled_del_sq = X_vars_i * del * del; 
            convg_measure = std::max<value_t>(convg_measure, scaled_del_sq);
            loss -= del * gi - 0.5 * scaled_del_sq;
            beta[i] = bi_new;
            resid -= del * X.col(i).array();
        }
        if (convg_measure < tol) break;
    }
}

} // namespace optimization
} // namespace adelie_core