#pragma once
#include <adelie_core/bcd/sgl/utils.hpp>

namespace adelie_core {
namespace bcd {
namespace sgl {
namespace unconstrained {

template <class ValueType>
void update_coordinate(
    ValueType& x_i,
    ValueType gamma,
    ValueType S,
    ValueType l1,
    ValueType l2,
    ValueType x_norm_sq_rest
)
{
    constexpr auto max_iters = 100000;
    constexpr auto tol = 1e-12;
    const auto abs_gamma = std::abs(gamma);
    x_i = (abs_gamma <= l1) ? 0 : std::copysign(
        root_secular(abs_gamma-l1, S, x_norm_sq_rest, l2, tol, max_iters), 
        gamma
    );
}

template <
    class SigmaType, 
    class VType, 
    class ValueType, 
    class XType, 
    class GradType
>
inline
void coordinate_descent_solver(
    const SigmaType& Sigma,
    const VType& v,
    ValueType l1,
    ValueType l2,
    ValueType tol,
    size_t max_iters,
    XType& x,
    GradType& grad,
    size_t& iters
)
{
    using value_t = ValueType;
    using sigma_t = SigmaType;

    iters = 0;

    // check if optimal solution is 0
    if ((v.abs() - l1).max(0).square().sum() <= l2 * l2) {
        x.setZero();
        grad = v;
        return;
    }

    // compute full objective f(x)
    const auto obj = (
        0.5 * (x.matrix() * Sigma).dot(x.matrix()) -
        (v * x).sum() + 
        l1 * x.matrix().template lpNorm<1>() + 
        l2 * x.matrix().norm()
    );

    // if f(x) >= f(0), set x to the default point where f(x) < f(0)
    if (obj >= 0) {
        x = (2 * (v > 0).template cast<value_t>() - 1) * (v.abs() - l1).max(0);
        const auto xTSx = (x.matrix() * Sigma).dot(x.matrix());
        const auto t = (
            (xTSx <= 0) ? 1 : (
                (x * v).sum() - 
                l1 * x.matrix().template lpNorm<1>() -
                l2 * x.matrix().norm()
            ) / xTSx
        );
        x *= t;
        grad = v.matrix() - x.matrix() * Sigma;
    }

    // vanilla coordinate descent
    const auto d = v.size();
    value_t x_norm_sq = x.matrix().squaredNorm();
    while (1) {
        ++iters;
        value_t convg_measure = 0;
        for (int i = 0; i < d; ++i) {
            const auto g_i = grad[i];
            const auto S_ii = Sigma(i, i);
            auto& x_i = x[i];
            const auto x_i_old = x_i;
            const auto gamma_i = g_i + S_ii * x_i;
            update_coordinate(
                x_i, 
                gamma_i, 
                S_ii, 
                l1, 
                l2, 
                std::max<value_t>(x_norm_sq - x_i_old * x_i_old, 0)
            );
            if (x_i == x_i_old) continue;
            const auto del = x_i - x_i_old;
            convg_measure = std::max(convg_measure, del * del * S_ii);
            if constexpr (sigma_t::IsRowMajor) {
                grad -= del * Sigma.row(i).array();
            } else{
                grad -= del * Sigma.col(i).array();
            }
            x_norm_sq += del * (x_i + x_i_old);
        }
        // Full SGL solver is still valid even if max_iters is reached.
        if (
            iters > max_iters || 
            convg_measure <= tol
        ) break;
    }
}

} // namespace unconstrained
} // namespace sgl
} // namespace bcd
} // namespace adelie_core