#pragma once
#include <adelie_core/util/macros.hpp>
#include <adelie_core/util/types.hpp>

namespace adelie_core {
namespace state {

template <class MatrixType,
          class ValueType=typename std::decay_t<MatrixType>::value_t,
          class IndexType=Eigen::Index,
          class BoolType=bool>
struct StateBVLS
{
    using matrix_t = MatrixType;
    using value_t = ValueType;
    using index_t = IndexType;
    using bool_t = BoolType;
    using vec_value_t = util::rowvec_type<value_t>;
    using vec_index_t = util::rowvec_type<index_t>;
    using vec_bool_t = util::rowvec_type<bool_t>;
    using map_vec_value_t = Eigen::Map<vec_value_t>;
    using map_vec_index_t = Eigen::Map<vec_index_t>;
    using map_vec_bool_t = Eigen::Map<vec_bool_t>;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;

    matrix_t* X;

    /* static states */
    const value_t y_var;
    const map_cvec_value_t X_vars;
    const map_cvec_value_t lower;
    const map_cvec_value_t upper;
    const map_cvec_value_t weights;

    /* configurations */
    const size_t kappa;
    const size_t max_iters;
    const value_t tol;

    /* dynamic states */
    size_t screen_set_size;
    map_vec_index_t screen_set;
    map_vec_bool_t is_screen;
    size_t active_set_size;
    map_vec_index_t active_set;
    map_vec_bool_t is_active;
    map_vec_value_t beta;
    map_vec_value_t resid;
    map_vec_value_t grad;
    value_t loss;
    size_t iters = 0;
    size_t n_kkt = 0;

    // debug
    std::vector<double> benchmark_fit_active;
    std::vector<double> benchmark_fit_screen;
    std::vector<double> benchmark_gradient;
    std::vector<double> benchmark_viols_sort;

    std::vector<vec_value_t> dbg_beta;
    std::vector<vec_index_t> dbg_active_set;
    std::vector<size_t> dbg_iter;
    std::vector<value_t> dbg_loss;

    virtual ~StateBVLS() =default;

    explicit StateBVLS(
        matrix_t& X,
        value_t y_var,
        const Eigen::Ref<const vec_value_t>& X_vars,
        const Eigen::Ref<const vec_value_t>& lower,
        const Eigen::Ref<const vec_value_t>& upper,
        const Eigen::Ref<const vec_value_t>& weights,
        size_t kappa,
        size_t max_iters,
        value_t tol,
        size_t screen_set_size,
        Eigen::Ref<vec_index_t> screen_set,
        Eigen::Ref<vec_bool_t> is_screen,
        size_t active_set_size,
        Eigen::Ref<vec_index_t> active_set,
        Eigen::Ref<vec_bool_t> is_active,
        Eigen::Ref<vec_value_t> beta,
        Eigen::Ref<vec_value_t> resid,
        Eigen::Ref<vec_value_t> grad,
        value_t loss
    ):
        X(&X),
        y_var(y_var),
        X_vars(X_vars.data(), X_vars.size()),
        lower(lower.data(), lower.size()),
        upper(upper.data(), upper.size()),
        weights(weights.data(), weights.size()),
        kappa(kappa),
        max_iters(max_iters),
        tol(tol),
        screen_set_size(screen_set_size),
        screen_set(screen_set.data(), screen_set.size()),
        is_screen(is_screen.data(), is_screen.size()),
        active_set_size(active_set_size),
        active_set(active_set.data(), active_set.size()),
        is_active(is_active.data(), is_active.size()),
        beta(beta.data(), beta.size()),
        resid(resid.data(), resid.size()),
        grad(grad.data(), grad.size()),
        loss(loss)
    {
        const auto n = X.rows();
        const auto p = X.cols();

        if (X_vars.size() != p) {
            throw util::adelie_core_solver_error(
                "X_vars must be (p,) where X is (n, p). "
            );
        }
        if (lower.size() != p) {
            throw util::adelie_core_solver_error(
                "lower must be (p,) where X is (n, p). "
            );
        }
        if (upper.size() != p) {
            throw util::adelie_core_solver_error(
                "upper must be (p,) where X is (n, p). "
            );
        }
        if (weights.size() != n) {
            throw util::adelie_core_solver_error(
                "weights must be (n,) where X is (n, p). "
            );
        }
        if (kappa <= 0) {
            throw util::adelie_core_solver_error(
                "kappa must be > 0. "
            );
        }
        if (tol < 0) {
            throw util::adelie_core_solver_error(
                "tol must be >= 0."
            );
        }
        if (static_cast<Eigen::Index>(active_set_size) > p) {
            throw util::adelie_core_solver_error(
                "active_set_size must be <= p where X is (n, p). "
            );
        }
        if (active_set.size() != p) {
            throw util::adelie_core_solver_error(
                "active_set must be (p,) where X is (n, p). "
            );
        }
        if (is_active.size() != p) {
            throw util::adelie_core_solver_error(
                "is_active must be (p,) where X is (n, p). "
            );
        }
        if (beta.size() != p) {
            throw util::adelie_core_solver_error(
                "beta must be (p,) where X is (p, n). "
            );
        }
        if (resid.size() != n) {
            throw util::adelie_core_solver_error(
                "resid must be (n,) where X is (n, p). "
            );
        }
        if (grad.size() != p) {
            throw util::adelie_core_solver_error(
                "grad must be (p,) where X is (n, p). "
            );
        }
    }
};

} // namespace state
} // namespace adelie_core