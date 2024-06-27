#pragma once
#include <adelie_core/util/types.hpp>

namespace adelie_core {
namespace optimization {

template <class MatrixType>
struct StateLassoFull
{
    using matrix_t = MatrixType;
    using value_t = typename std::decay_t<MatrixType>::Scalar;
    using vec_value_t = util::rowvec_type<value_t>;
    using map_vec_value_t = Eigen::Map<vec_value_t>;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;
    using map_cmatrix_t = Eigen::Map<const matrix_t>;

    const map_cmatrix_t quad;
    const map_cvec_value_t penalty;

    const size_t max_iters;
    const value_t tol;

    size_t iters = 0;
    map_vec_value_t x;      
    map_vec_value_t grad;

    double time_elapsed = 0;

    explicit StateLassoFull(
        const Eigen::Ref<const matrix_t>& quad,
        const Eigen::Ref<const vec_value_t>& penalty,
        size_t max_iters,
        value_t tol,
        Eigen::Ref<vec_value_t> x,
        Eigen::Ref<vec_value_t> grad
    ):
        quad(quad.data(), quad.rows(), quad.cols()),
        penalty(penalty.data(), penalty.size()),
        max_iters(max_iters),
        tol(tol),
        x(x.data(), x.size()),
        grad(grad.data(), grad.size())
    {}
};

template <class StateType>
void lasso_full(
    StateType& state
)
{
    using state_t = std::decay_t<StateType>;
    using matrix_t = typename state_t::matrix_t;
    using value_t = typename state_t::value_t;

    const auto& quad = state.quad;
    const auto& penalty = state.penalty;
    const auto max_iters = state.max_iters;
    const auto tol = state.tol;
    auto& iters = state.iters;
    auto& x = state.x;
    auto& grad = state.grad;

    const auto n = x.size();

    iters = 0;

    while (iters < max_iters) {
        value_t convg_measure = 0;
        ++iters;
        for (int i = 0; i < n; ++i) {
            const auto qii = quad(i,i);
            const auto pi = penalty[i];
            auto& xi = x[i];
            const auto gi = grad[i];
            const auto xi_old = xi;
            const auto gi0 = gi + qii * xi_old;
            const auto gi0_abs = std::abs(gi0);
            xi = (gi0_abs <= pi) ? 0 : std::copysign((gi0_abs - pi) / qii, gi0);
            const auto del = xi - xi_old;
            if (del == 0) continue;
            const auto scaled_del_sq = qii * del * del; 
            convg_measure = std::max<value_t>(convg_measure, scaled_del_sq);
            if constexpr (matrix_t::IsRowMajor) {
                grad -= del * quad.array().row(i);
            } else {
                grad -= del * quad.array().col(i);
            }
        }
        if (convg_measure < tol) break;
    }
}

} // namespace optimization
} // namespace adelie_core