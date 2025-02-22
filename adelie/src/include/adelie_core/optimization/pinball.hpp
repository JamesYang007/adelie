#pragma once
#include <adelie_core/util/types.hpp>
#ifdef ADELIE_CORE_DEBUG
#undef ADELIE_CORE_DEBUG
#define _ADELIE_CORE_DEBUG
#endif
#include <adelie_core/solver/solver_pinball.hpp>

namespace adelie_core {
namespace optimization {

template <
    class MatrixType,
    class ValueType=typename std::decay_t<MatrixType>::value_t,
    class IndexType=Eigen::Index,
    class BoolType=bool
>
struct StatePinball
{
    using matrix_t = MatrixType;
    using value_t = ValueType;
    using index_t = IndexType;
    using bool_t = BoolType;
    using vec_value_t = util::rowvec_type<value_t>;
    using vec_index_t = util::rowvec_type<index_t>;
    using vec_bool_t = util::rowvec_type<bool_t>;
    using colmat_value_t = util::colmat_type<value_t>;
    using rowmat_value_t = util::rowmat_type<value_t>;
    using map_vec_value_t = Eigen::Map<vec_value_t>;
    using map_vec_index_t = Eigen::Map<vec_index_t>;
    using map_vec_bool_t = Eigen::Map<vec_bool_t>;
    using map_rowmat_value_t = Eigen::Map<rowmat_value_t>;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;
    using map_ccolmat_value_t = Eigen::Map<const colmat_value_t>;

    matrix_t* A;

    /* static states */
    const value_t y_var;
    const map_ccolmat_value_t S;
    const map_cvec_value_t penalty_neg;
    const map_cvec_value_t penalty_pos;

    /* configurations */
    const size_t kappa;
    const size_t max_iters;
    const value_t tol;

    /* dynamic states */
    size_t screen_set_size;
    map_vec_index_t screen_set;
    map_vec_bool_t is_screen;
    map_vec_value_t screen_ASAT_diag;
    map_rowmat_value_t screen_AS;
    size_t active_set_size;
    map_vec_index_t active_set;
    map_vec_bool_t is_active;
    map_vec_value_t beta;
    map_vec_value_t resid;
    map_vec_value_t grad;
    value_t loss;
    size_t iters = 0;
    size_t n_kkt = 0;

    double time_elapsed = 0;

    explicit StatePinball(
        matrix_t& A,
        value_t y_var,
        const Eigen::Ref<const colmat_value_t>& S,
        const Eigen::Ref<const vec_value_t>& penalty_neg,
        const Eigen::Ref<const vec_value_t>& penalty_pos,
        size_t kappa,
        size_t max_iters,
        value_t tol,
        size_t screen_set_size,
        Eigen::Ref<vec_index_t> screen_set,
        Eigen::Ref<vec_bool_t> is_screen,
        Eigen::Ref<vec_value_t> screen_ASAT_diag,
        Eigen::Ref<rowmat_value_t> screen_AS,
        size_t active_set_size,
        Eigen::Ref<vec_index_t> active_set,
        Eigen::Ref<vec_bool_t> is_active,
        Eigen::Ref<vec_value_t> beta,
        Eigen::Ref<vec_value_t> resid,
        Eigen::Ref<vec_value_t> grad,
        value_t loss
    ):
        A(&A),
        y_var(y_var),
        S(S.data(), S.rows(), S.cols()),
        penalty_neg(penalty_neg.data(), penalty_neg.size()),
        penalty_pos(penalty_pos.data(), penalty_pos.size()),
        kappa(kappa),
        max_iters(max_iters),
        tol(tol),
        screen_set_size(screen_set_size),
        screen_set(screen_set.data(), screen_set.size()),
        is_screen(is_screen.data(), is_screen.size()),
        screen_ASAT_diag(screen_ASAT_diag.data(), screen_ASAT_diag.size()),
        screen_AS(screen_AS.data(), screen_AS.rows(), screen_AS.cols()),
        active_set_size(active_set_size),
        active_set(active_set.data(), active_set.size()),
        is_active(is_active.data(), is_active.size()),
        beta(beta.data(), beta.size()),
        resid(resid.data(), resid.size()),
        grad(grad.data(), grad.size()),
        loss(loss)
    {
        const auto m = A.rows();
        const auto d = A.cols();

        if (S.rows() != d || S.cols() != d) {
            throw util::adelie_core_solver_error(
                "S must be (d, d) where A is (m, d). "
            );
        }
        if (penalty_neg.size() != m) {
            throw util::adelie_core_solver_error(
                "penalty_neg must be (m,) where A is (m, d). "
            );
        }
        if (penalty_pos.size() != m) {
            throw util::adelie_core_solver_error(
                "penalty_pos must be (m,) where A is (m, d). "
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
        if (static_cast<Eigen::Index>(screen_set_size) > m) {
            throw util::adelie_core_solver_error(
                "screen_set_size must be <= m where A is (m, d). "
            );
        }
        if (screen_set.size() != m) {
            throw util::adelie_core_solver_error(
                "screen_set must be (m,) where A is (m, d). "
            );
        }
        if (is_screen.size() != m) {
            throw util::adelie_core_solver_error(
                "is_screen must be (m,) where A is (m, d). "
            );
        }
        if (screen_ASAT_diag.size() != m) {
            throw util::adelie_core_solver_error(
                "screen_ASAT_diag must be (m,) where A is (m, d). "
            );
        }
        if (screen_AS.rows() != m || screen_AS.cols() != d) {
            throw util::adelie_core_solver_error(
                "screen_AS must be (m, d) where A is (m, d). "
            );
        }
        if (static_cast<Eigen::Index>(active_set_size) > m) {
            throw util::adelie_core_solver_error(
                "active_set_size must be <= m where A is (m, d). "
            );
        }
        if (active_set.size() != m) {
            throw util::adelie_core_solver_error(
                "active_set must be (m,) where A is (m, d). "
            );
        }
        if (is_active.size() != m) {
            throw util::adelie_core_solver_error(
                "is_active must be (m,) where A is (m, d). "
            );
        }
        if (beta.size() != m) {
            throw util::adelie_core_solver_error(
                "beta must be (m,) where A is (m, d). "
            );
        }
        if (resid.size() != d) {
            throw util::adelie_core_solver_error(
                "resid must be (d,) where A is (m, d). "
            );
        }
        if (grad.size() != m) {
            throw util::adelie_core_solver_error(
                "grad must be (m,) where A is (m, d). "
            );
        }
    }

    void solve()
    {
        solver::pinball::solve(*this); 
    }
};

} // namespace optimization
} // namespace adelie_core

#ifdef _ADELIE_CORE_DEBUG
#undef _ADELIE_CORE_DEBUG
#define ADELIE_CORE_DEBUG
#endif