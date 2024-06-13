#pragma once
#include <adelie_core/util/types.hpp>

namespace adelie_core {
namespace optimization {

template <class MatrixType,
          class ValueType=typename std::decay_t<MatrixType>::Scalar>
struct StateNNQPFull
{
    using matrix_t = MatrixType;
    using value_t = ValueType;
    using vec_value_t = util::rowvec_type<value_t>;
    using map_vec_value_t = Eigen::Map<vec_value_t>;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;
    using map_cmatrix_t = Eigen::Map<const matrix_t>;

    const map_cmatrix_t quad;

    const size_t max_iters;
    const value_t tol;
    const value_t dtol;

    size_t iters = 0;
    map_vec_value_t x;      
    map_vec_value_t grad;

    double time_elapsed = 0;

    explicit StateNNQPFull(
        const Eigen::Ref<const matrix_t>& quad,
        size_t max_iters,
        value_t tol,
        value_t dtol,
        Eigen::Ref<vec_value_t> x,
        Eigen::Ref<vec_value_t> grad
    ):
        quad(quad.data(), quad.rows(), quad.cols()),
        max_iters(max_iters),
        tol(tol),
        dtol(dtol),
        x(x.data(), x.size()),
        grad(grad.data(), grad.size())
    {}
};

template <class StateType>
void nnqp_full(
    StateType& state
)
{
    using state_t = std::decay_t<StateType>;
    using matrix_t = typename state_t::matrix_t;
    using value_t = typename state_t::value_t;

    const auto& quad = state.quad;
    const auto max_iters = state.max_iters;
    const auto tol = state.tol;
    const auto dtol = state.dtol;
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
            auto& xi = x[i];
            if (qii <= 0) { 
                xi = std::max<value_t>(xi, 0); 
                continue;
            }
            const auto gi = grad[i];
            const auto xi_old = xi;
            xi = std::max<value_t>(xi + gi / qii, 0);
            const auto del = xi - xi_old;
            if (std::abs(del) <= dtol) continue;
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