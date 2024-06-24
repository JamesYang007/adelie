#pragma once
#include <adelie_core/util/types.hpp>

namespace adelie_core {
namespace optimization {

template <class MatrixType,
          class ValueType=typename std::decay_t<MatrixType>::Scalar>
struct StateNNLS
{
    using matrix_t = MatrixType;
    using value_t = ValueType;
    using vec_value_t = util::rowvec_type<value_t>;
    using map_vec_value_t = Eigen::Map<vec_value_t>;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;
    using map_cmatrix_t = Eigen::Map<const matrix_t>;

    const map_cmatrix_t X;
    const map_cvec_value_t X_vars;

    const size_t max_iters;
    const value_t tol;

    size_t iters = 0;
    map_vec_value_t beta;      
    map_vec_value_t resid;
    value_t loss;

    double time_elapsed = 0;

    StateNNLS(
        const Eigen::Ref<const matrix_t>& X,
        const Eigen::Ref<const vec_value_t>& X_vars,
        size_t max_iters,
        value_t tol,
        Eigen::Ref<vec_value_t> beta,
        Eigen::Ref<vec_value_t> resid,
        value_t loss
    ):
        X(X.data(), X.rows(), X.cols()),
        X_vars(X_vars.data(), X_vars.size()),
        max_iters(max_iters),
        tol(tol),
        beta(beta.data(), beta.size()),
        resid(resid.data(), resid.size()),
        loss(loss)
    {}
};

template <class StateType, class EarlyExitType, class SkipType>
void nnls(
    StateType& state,
    EarlyExitType early_exit_f,
    SkipType skip_f
)
{
    using state_t = std::decay_t<StateType>;
    using value_t = typename state_t::value_t;

    const auto& X = state.X;
    const auto& X_vars = state.X_vars;
    const auto max_iters = state.max_iters;
    const auto tol = state.tol;
    auto& iters = state.iters;
    auto& beta = state.beta;
    auto& resid = state.resid;
    auto& loss = state.loss;

    const auto n = beta.size();

    iters = 0;

    while (iters < max_iters) {
        value_t convg_measure = 0;
        ++iters;
        for (int i = 0; i < n; ++i) {
            if (early_exit_f()) return;
            if (skip_f(i)) continue;
            const auto X_vars_i = X_vars[i];
            auto& bi = beta[i];
            const auto gi = X.col(i).dot(resid.matrix());
            const auto bi_old = bi;
            const auto step = (X_vars_i <= 0) ? 0 : (gi / X_vars_i);
            bi = std::max<value_t>(bi + step, 0.0);
            const auto del = bi - bi_old;
            if (del == 0) continue;
            const auto scaled_del_sq = X_vars_i * del * del; 
            convg_measure = std::max<value_t>(convg_measure, scaled_del_sq);
            loss -= del * gi - 0.5 * scaled_del_sq;
            resid -= del * X.col(i).array();
        }
        if (convg_measure < tol) break;
    }
}

} // namespace optimization
} // namespace adelie_core