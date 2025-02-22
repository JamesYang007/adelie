#pragma once
#include <adelie_core/configs.hpp>
#include <adelie_core/constraint/constraint_linear.hpp>
#include <adelie_core/constraint/utils.hpp>
#include <adelie_core/optimization/nnls.hpp>
#include <adelie_core/optimization/pinball.hpp>
#include <adelie_core/optimization/pinball_full.hpp>
#include <adelie_core/util/omp.hpp>

namespace adelie_core {
namespace constraint { 
namespace linear {

template <class MatrixType>
struct MatrixConstraintNNLS
{
    using matrix_t = MatrixType;
    using value_t = typename matrix_t::value_t;
    using vec_value_t = typename matrix_t::vec_value_t;

    matrix_t* _A;

    MatrixConstraintNNLS(
        matrix_t& A
    ):
        _A(&A)
    {}

    value_t cmul(
        int j, 
        const Eigen::Ref<const vec_value_t>& v,
        const Eigen::Ref<const vec_value_t>& 
    ) 
    {
        return _A->rvmul(j, v);
    }

    void ctmul(
        int j, 
        value_t v, 
        Eigen::Ref<vec_value_t> out
    ) 
    {
        _A->rvtmul(j, v, out);
    }

    void mul(
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>&,
        Eigen::Ref<vec_value_t> out
    ) const 
    {
        _A->tmul(v, out);
    }

    int rows() const 
    {
        return _A->cols();
    }
    
    int cols() const
    {
        return _A->rows();
    }
};

} // namespace linear

ADELIE_CORE_CONSTRAINT_LINEAR_TP
void
ADELIE_CORE_CONSTRAINT_LINEAR::compute_ATmu(
    Eigen::Ref<vec_value_t> out
) 
{
    const Eigen::Map<const vec_index_t> indices(_mu_active.data(), _mu_active.size());
    const Eigen::Map<const vec_value_t> values(_mu_value.data(), _mu_value.size());
    _A->sp_mul(indices, values, out);
};

ADELIE_CORE_CONSTRAINT_LINEAR_TP
void
ADELIE_CORE_CONSTRAINT_LINEAR::mu_to_dense(
    Eigen::Ref<vec_value_t> mu
) 
{
    mu.setZero();
    for (size_t i = 0; i < _mu_active.size(); ++i) {
        mu[_mu_active[i]] = _mu_value[i]; 
    }
};

ADELIE_CORE_CONSTRAINT_LINEAR_TP
void
ADELIE_CORE_CONSTRAINT_LINEAR::mu_to_sparse(
    Eigen::Ref<vec_value_t> mu
) 
{
    for (size_t i = 0; i < _mu_active.size(); ++i) {
        _mu_value[i] = mu[_mu_active[i]];
    }
    for (Eigen::Index i = 0; i < mu.size(); ++i) {
        const auto mi = mu[i];
        if (mi == 0 || _mu_active_set.find(i) != _mu_active_set.end()) continue;
        _mu_active_set.insert(i);
        _mu_active.push_back(i);
        _mu_value.push_back(mi);
    }
    mu_prune(_eps);
};

ADELIE_CORE_CONSTRAINT_LINEAR_TP
void
ADELIE_CORE_CONSTRAINT_LINEAR::mu_prune(value_t eps) 
{
    size_t n_active = 0;
    for (size_t i = 0; i < _mu_active.size(); ++i) {
        const auto idx = _mu_active[i];
        const auto mi = _mu_value[i];
        if (std::abs(mi) <= eps) {
            _mu_active_set.erase(idx);
            continue;
        }
        _mu_active[n_active] = idx;
        _mu_value[n_active] = mi;
        ++n_active;
    }
    _mu_active.erase(
        std::next(_mu_active.begin(), n_active),
        _mu_active.end()
    );
    _mu_value.erase(
        std::next(_mu_value.begin(), n_active),
        _mu_value.end()
    );
}

ADELIE_CORE_CONSTRAINT_LINEAR_TP
void
ADELIE_CORE_CONSTRAINT_LINEAR::_clear()
{
    _mu_active_set.clear();
    _mu_active.clear();
    _mu_value.clear();
    _ATmu.setZero();
}

ADELIE_CORE_CONSTRAINT_LINEAR_TP
ADELIE_CORE_CONSTRAINT_LINEAR::ConstraintLinear(
    A_t& A,
    const Eigen::Ref<const vec_value_t>& l,
    const Eigen::Ref<const vec_value_t>& u,
    const Eigen::Ref<const vec_value_t>& A_vars,
    size_t max_iters,
    value_t tol,
    size_t nnls_max_iters,
    value_t nnls_tol,
    size_t pinball_max_iters,
    value_t pinball_tol,
    value_t slack,
    size_t n_threads
):
    _A(&A),
    _l(l.data(), l.size()),
    _u(u.data(), u.size()),
    _A_vars(A_vars.data(), A_vars.size()),
    _max_iters(max_iters),
    _tol(tol),
    _nnls_max_iters(nnls_max_iters),
    _nnls_tol(nnls_tol),
    _pinball_max_iters(pinball_max_iters),
    _pinball_tol(pinball_tol),
    _slack(slack),
    _n_threads(n_threads),
    _ATmu(vec_value_t::Zero(A.cols()))
{
    const auto m = A.rows();

    if (l.size() != m) {
        throw util::adelie_core_error("lower must be (m,) where A is (m, d).");
    }
    if (u.size() != m) {
        throw util::adelie_core_error("upper must be (m,) where A is (m, d).");
    }
    if ((u < 0).any()) {
        throw util::adelie_core_error("upper must be >= 0.");
    }
    if ((l < 0).any()) { 
        // NOTE: user passes in lower == -l
        throw util::adelie_core_error("lower must be <= 0.");
    }
    if (A_vars.size() != m) {
        throw util::adelie_core_error("A_vars must be (m,) where A is (m, d).");
    }
    if (tol < 0) {
        throw util::adelie_core_error("tol must be >= 0.");
    }
    if (nnls_tol < 0) {
        throw util::adelie_core_error("nnls_tol must be >= 0.");
    }
    if (pinball_tol < 0) {
        throw util::adelie_core_error("pinball_tol must be >= 0.");
    }
    if (slack <= 0 || slack >= 1) {
        throw util::adelie_core_error("slack must be in (0,1).");
    }
}

ADELIE_CORE_CONSTRAINT_LINEAR_TP
void
ADELIE_CORE_CONSTRAINT_LINEAR::project(
    Eigen::Ref<vec_value_t>
) const
{}

ADELIE_CORE_CONSTRAINT_LINEAR_TP
size_t
ADELIE_CORE_CONSTRAINT_LINEAR::buffer_size() const 
{
    const auto m = _A->rows();
    const auto d = _A->cols();
    return d * (9 + 2 * d) + 5 * m + ((m < d) ? (m * m) : ((1 + d) * m));
}

ADELIE_CORE_CONSTRAINT_LINEAR_TP
void
ADELIE_CORE_CONSTRAINT_LINEAR::solve(
    Eigen::Ref<vec_value_t> x,
    const Eigen::Ref<const vec_value_t>& quad,
    const Eigen::Ref<const vec_value_t>& linear,
    value_t l1,
    value_t l2,
    const Eigen::Ref<const colmat_value_t>& Q,
    Eigen::Ref<vec_uint64_t> buffer
) 
{
    using vec_bool_t = util::rowvec_type<bool_t>;
    using internal_matrix_t = linear::MatrixConstraintNNLS<A_t>;

    const auto m = _A->rows();
    const auto d = _A->cols();

    base_t::check_solve(x.size(), quad.cols(), linear.size(), m, d);

    // check if x = 0, mu = 0 is optimal
    const auto v_norm = linear.matrix().norm();
    if (v_norm <= l1) {
        x.setZero();
        _clear(); 
        return;
    }

    auto buff_iptr = reinterpret_cast<index_t*>(buffer.data());
    const auto buff_ibegin = buff_iptr;
    Eigen::Map<vec_index_t> screen_set(buff_iptr, m); buff_iptr += m;
    Eigen::Map<vec_index_t> active_set(buff_iptr, m); buff_iptr += m;
    size_t n_read = std::distance(buff_ibegin, buff_iptr);

    auto buff_vptr = reinterpret_cast<value_t*>(buffer.data() + n_read);
    const auto buff_vbegin = buff_vptr;
    Eigen::Map<vec_value_t> grad_prev(buff_vptr, d); buff_vptr += d;
    Eigen::Map<vec_value_t> grad(buff_vptr, d); buff_vptr += d;
    Eigen::Map<vec_value_t> ATmu_prev(buff_vptr, d); buff_vptr += d;
    Eigen::Map<vec_value_t> mu(buff_vptr, m); buff_vptr += m;
    Eigen::Map<vec_value_t> pinball_grad(buff_vptr, m); buff_vptr += m;
    Eigen::Map<vec_value_t> nnls_grad(buff_vptr, m); buff_vptr += m;
    const auto m_large = (m < d) ? 0 : m;
    Eigen::Map<vec_value_t> screen_ASAT_diag(buff_vptr, m_large); buff_vptr += m_large;
    Eigen::Map<rowmat_value_t> screen_AS(buff_vptr, m_large, d); buff_vptr += m_large * d;
    const auto m_small = (m < d) ? m : 0;
    Eigen::Map<colmat_value_t> hess_small(buff_vptr, m_small, m_small); buff_vptr += m_small * m_small;
    n_read += std::distance(buff_vbegin, buff_vptr);
    Eigen::Map<vec_uint64_t> next_buff(buffer.data() + n_read, buffer.size() - n_read);

    const auto compute_mu_resid = [&](
        auto& mu_resid
    ) {
        mu_resid.matrix() = linear.matrix() - _ATmu.matrix() * Q;
    };
    const auto compute_min_mu_resid = [&](
        const auto& Qv,
        bool is_prev_valid_old,
        bool is_init
    ) {
        // check if current mu_resid norm is small enough
        if ((Qv - _ATmu).square().sum() <= l1 * l1) return value_t(0);

        const auto lower_constraint = vec_value_t::NullaryExpr(_l.size(), [&](auto i) {
            const auto li = _l[i];
            return (li <= 0) ? (-Configs::max_solver_value) : 0;
        });
        const auto upper_constraint = vec_value_t::NullaryExpr(_u.size(), [&](auto i) {
            const auto ui = _u[i];
            return (ui <= 0) ? Configs::max_solver_value : 0;
        });

        Eigen::Map<vec_bool_t> is_screen(reinterpret_cast<bool_t*>(pinball_grad.data()), m);
        Eigen::Map<vec_bool_t> is_active(reinterpret_cast<bool_t*>(pinball_grad.data())+m, m);
        auto& Qmu_resid = grad;

        is_screen.setZero();
        is_active.setZero();
        mu.setZero();
        for (size_t i = 0; i < _mu_active.size(); ++i) {
            const auto idx = _mu_active[i];
            const auto val = _mu_value[i];
            is_screen[idx] = true;
            is_active[idx] = true;
            screen_set[i] = idx;
            active_set[i] = idx;
            mu[idx] = val;
        }
        Qmu_resid = Qv - _ATmu;
        const value_t loss = 0.5 * Qmu_resid.square().sum();
        internal_matrix_t _X(*_A); // _X == _A^T
        optimization::StateNNLS<internal_matrix_t, value_t, index_t> state_nnls(
            _X, v_norm * v_norm, _A_vars, std::min<size_t>(m, d),
            _nnls_max_iters, _nnls_tol, 
            _mu_active.size(),
            screen_set,
            is_screen,
            _mu_active.size(),
            active_set, 
            is_active, 
            mu, Qmu_resid, nnls_grad, loss
        );
        state_nnls.solve(
            [&]() { return 2 * state_nnls.loss <= l1 * l1; },
            lower_constraint,
            upper_constraint
        );
        const value_t mu_resid_norm_sq = 2 * state_nnls.loss;

        if ((!is_init && !is_prev_valid_old) || (mu_resid_norm_sq <= l1 * l1)) {
            _mu_active.clear();
            _mu_value.clear();
            for (size_t i = 0; i < state_nnls.active_set_size; ++i) {
                const auto idx = active_set[i];
                const auto val = mu[idx];
                _mu_active.push_back(idx);
                _mu_value.push_back(val);
            }
            _mu_active_set.clear();
            _mu_active_set.insert(
                _mu_active.data(),
                _mu_active.data() + _mu_active.size()
            );
            mu_prune(_eps);
            _ATmu = Qv - Qmu_resid;
        }

        return mu_resid_norm_sq;
    };
    const auto compute_backtrack_a = [&]() {
        const auto ATdmu = _ATmu - ATmu_prev;
        return ATdmu.square().sum();
    };
    const auto compute_backtrack_b = [&](
        const auto&,
        const auto& mu_resid
    ) {
        const auto ATdmu = _ATmu - ATmu_prev;
        return (mu_resid.matrix() * Q.transpose()).dot(ATdmu.matrix());
    };
    const auto compute_backtrack = [&](auto step_size) {
        for (size_t i = 0; i < _mu_active_prev.size(); ++i) {
            mu[_mu_active_prev[i]] = (1-step_size) * _mu_value_prev[i];
        }
        for (size_t i = 0; i < _mu_active.size(); ++i) {
            const auto idx = _mu_active[i];
            auto& mi = _mu_value[i];
            mi = step_size * mi + (
                (_mu_active_set_prev.find(idx) != _mu_active_set_prev.end()) ?
                mu[idx] : 0
            );
        }
        for (size_t i = 0; i < _mu_active_prev.size(); ++i) {
            const auto idx = _mu_active_prev[i];
            if (_mu_active_set.find(idx) != _mu_active_set.end()) continue;
            _mu_active_set.insert(idx);
            _mu_active.push_back(idx);
            _mu_value.push_back(mu[idx]);
        }
        compute_ATmu(_ATmu);
    };
    const auto compute_gradient = [&]() {
        grad.matrix() = x.matrix() * Q.transpose();
    };
    const auto compute_hard_optimality = [&]() {
        // This is an optional check for optimization purposes.
        // The cost of computing this check may be too much if m >> d
        // so we omit this check.
        return false;
    };
    const auto compute_convergence_measure = [&](
        bool is_in_ellipse
    ) {
        const auto ATdmu = (_ATmu - ATmu_prev);
        return is_in_ellipse ? (
            std::abs((ATdmu * grad_prev).mean())
        ) : (
            std::abs((ATdmu * (grad_prev-grad)).mean())
        );
    };
    const auto compute_proximal_newton_step = [&](
        const auto& hess,
        const auto var
    ) {
        if (m < d) {
            mu_to_dense(mu);
            _A->cov(hess, hess_small);
            _A->tmul(grad, pinball_grad);
            optimization::StatePinballFull<colmat_value_t> state_pinball(
                hess_small, _l, _u, var, _pinball_max_iters, _pinball_tol,
                mu, pinball_grad
            );
            state_pinball.solve();
            mu_to_sparse(mu);
        } else {
            Eigen::Map<vec_bool_t> is_screen(reinterpret_cast<bool_t*>(nnls_grad.data()), m);
            Eigen::Map<vec_bool_t> is_active(reinterpret_cast<bool_t*>(nnls_grad.data())+m, m);
            auto& resid = grad;

            is_screen.setZero();
            is_active.setZero();
            mu.setZero();
            for (size_t i = 0; i < _mu_active.size(); ++i) {
                const auto idx = _mu_active[i];
                const auto val = _mu_value[i];
                screen_set[i] = idx;
                active_set[i] = idx;
                is_screen[idx] = true;
                is_active[idx] = true;
                mu[idx] = val;
            }

            const auto screen_invariance = [&](auto ii) {
                const auto i = _mu_active[ii];
                auto AS_i = screen_AS.row(i);
                _A->rmmul_safe(i, hess, AS_i);
                screen_ASAT_diag[i] = std::max<value_t>(_A->rvmul_safe(i, AS_i), 0);
            };
            const size_t active_size = _mu_active.size();
            const size_t n_bytes = sizeof(value_t) * d * (d + 1) * active_size;
            util::omp_parallel_for(screen_invariance, 0, active_size, _n_threads * (n_bytes > Configs::min_bytes));

            optimization::StatePinball<A_t, value_t, index_t> state_pinball(
                *_A, var, hess, _l, _u, 
                std::min(m, d), _pinball_max_iters, _pinball_tol, 
                _mu_active.size(),
                screen_set,
                is_screen,
                screen_ASAT_diag,
                screen_AS,
                _mu_active.size(),
                active_set,
                is_active,
                mu,
                resid,
                pinball_grad,
                0 /* loss is only used as a relative difference internally */
            );
            //using sw_t = util::Stopwatch;
            //sw_t sw;
            //sw.start();
            state_pinball.solve();
            //const auto elapsed = sw.elapsed();
            //PRINT(this);
            //PRINT(elapsed);
            //PRINT(state_pinball.iters);

            _mu_active.clear();
            _mu_value.clear();
            for (size_t i = 0; i < state_pinball.active_set_size; ++i) {
                const auto idx = active_set[i];
                const auto val = mu[idx];
                _mu_active.push_back(idx);
                _mu_value.push_back(val);
            }
            _mu_active_set.clear();
            _mu_active_set.insert(
                _mu_active.data(),
                _mu_active.data() + _mu_active.size()
            );
            mu_prune(_eps);
        }
        compute_ATmu(_ATmu);
    };
    const auto save_additional_prev = [&](bool is_in_ellipse) {
        _mu_active_set_prev = _mu_active_set;
        _mu_active_prev = _mu_active;
        _mu_value_prev = _mu_value;
        ATmu_prev = _ATmu;
        if (is_in_ellipse) grad_prev.setZero();
        else grad_prev = grad;
    };

    solve_proximal_newton(
        x, quad, linear, l1, l2, Q, _max_iters, _tol, _slack, next_buff,
        compute_mu_resid,
        compute_min_mu_resid,
        compute_backtrack_a,
        compute_backtrack_b,
        compute_backtrack,
        compute_gradient,
        compute_hard_optimality,
        compute_convergence_measure,
        compute_proximal_newton_step,
        save_additional_prev
    );
}

ADELIE_CORE_CONSTRAINT_LINEAR_TP
void
ADELIE_CORE_CONSTRAINT_LINEAR::gradient(
    const Eigen::Ref<const vec_value_t>&,
    Eigen::Ref<vec_value_t> out
) const
{
    out = _ATmu;
}

ADELIE_CORE_CONSTRAINT_LINEAR_TP
void
ADELIE_CORE_CONSTRAINT_LINEAR::gradient(
    const Eigen::Ref<const vec_value_t>&,
    const Eigen::Ref<const vec_value_t>& mu,
    Eigen::Ref<vec_value_t> out
) const
{
    _A->mul(mu, out);
}

ADELIE_CORE_CONSTRAINT_LINEAR_TP
typename ADELIE_CORE_CONSTRAINT_LINEAR::value_t
ADELIE_CORE_CONSTRAINT_LINEAR::solve_zero(
    const Eigen::Ref<const vec_value_t>& v,
    Eigen::Ref<vec_uint64_t> buffer
) 
{
    using vec_bool_t = util::rowvec_type<bool_t>;
    using internal_matrix_t = linear::MatrixConstraintNNLS<A_t>;

    const auto m = _A->rows();
    const auto d = _A->cols();

    auto buff_iptr = reinterpret_cast<index_t*>(buffer.data());
    const auto buff_ibegin = buff_iptr;
    Eigen::Map<vec_index_t> screen_set(buff_iptr, m); buff_iptr += m;
    Eigen::Map<vec_index_t> active_set(buff_iptr, m); buff_iptr += m;
    size_t n_read = std::distance(buff_ibegin, buff_iptr);

    auto buff_vptr = reinterpret_cast<value_t*>(buffer.data() + n_read);
    Eigen::Map<vec_value_t> grad(buff_vptr, d); buff_vptr += d;
    Eigen::Map<vec_value_t> mu(buff_vptr, m); buff_vptr += m;
    Eigen::Map<vec_value_t> nnls_grad(buff_vptr, m); buff_vptr += m;
    Eigen::Map<vec_bool_t> is_screen(reinterpret_cast<bool_t*>(buff_vptr), m);
    Eigen::Map<vec_bool_t> is_active(reinterpret_cast<bool_t*>(buff_vptr)+m, m); buff_vptr += m;

    is_screen.setZero();
    is_active.setZero();
    mu.setZero();
    for (size_t i = 0; i < _mu_active.size(); ++i) {
        const auto idx = _mu_active[i];
        const auto val = _mu_value[i];
        screen_set[i] = idx;
        active_set[i] = idx;
        is_screen[idx] = true;
        is_active[idx] = true;
        mu[idx] = val;
    }

    auto& Qmu_resid = grad;
    Qmu_resid = v - _ATmu;
    const value_t loss = 0.5 * Qmu_resid.square().sum();
    const auto lower_constraint = vec_value_t::NullaryExpr(_l.size(), [&](auto i) {
        const auto li = _l[i];
        return (li <= 0) ? (-Configs::max_solver_value) : 0;
    });
    const auto upper_constraint = vec_value_t::NullaryExpr(_u.size(), [&](auto i) {
        const auto ui = _u[i];
        return (ui <= 0) ? Configs::max_solver_value : 0;
    });
    internal_matrix_t _X(*_A); // _X == _A^T
    optimization::StateNNLS<internal_matrix_t, value_t, index_t> state_nnls(
        _X, v.square().sum(), _A_vars, std::min<size_t>(m, d), 
        _nnls_max_iters, _nnls_tol, 
        _mu_active.size(),
        screen_set,
        is_screen,
        _mu_active.size(),
        active_set,
        is_active,
        mu, Qmu_resid, nnls_grad, loss
    );
    state_nnls.solve(
        [&]() { return false; },
        lower_constraint,
        upper_constraint
    );

    _mu_active.clear();
    _mu_value.clear();
    for (size_t i = 0; i < state_nnls.active_set_size; ++i) {
        const auto idx = active_set[i];
        const auto val = mu[idx];
        _mu_active.push_back(idx);
        _mu_value.push_back(val);
    }
    _mu_active_set.clear();
    _mu_active_set.insert(
        _mu_active.data(),
        _mu_active.data() + _mu_active.size()
    );
    mu_prune(_eps);
    _ATmu = v - Qmu_resid;

    return std::sqrt(std::max<value_t>(2 * state_nnls.loss, 0));
}

ADELIE_CORE_CONSTRAINT_LINEAR_TP
void
ADELIE_CORE_CONSTRAINT_LINEAR::clear() 
{
    _clear();
}

ADELIE_CORE_CONSTRAINT_LINEAR_TP
void
ADELIE_CORE_CONSTRAINT_LINEAR::dual(
    Eigen::Ref<vec_index_t> indices,
    Eigen::Ref<vec_value_t> values
) const
{
    const size_t nnz = _mu_active.size();
    indices.head(nnz) = Eigen::Map<const vec_index_t>(
        _mu_active.data(),
        nnz
    );
    values.head(nnz) = Eigen::Map<const vec_value_t>(
        _mu_value.data(),
        nnz
    );
}

ADELIE_CORE_CONSTRAINT_LINEAR_TP
int
ADELIE_CORE_CONSTRAINT_LINEAR::duals_nnz() const 
{
    return _mu_active.size();
}

ADELIE_CORE_CONSTRAINT_LINEAR_TP
int
ADELIE_CORE_CONSTRAINT_LINEAR::duals() const 
{ 
    return _A->rows(); 
}

ADELIE_CORE_CONSTRAINT_LINEAR_TP
int
ADELIE_CORE_CONSTRAINT_LINEAR::primals() const 
{ 
    return _A->cols(); 
}

} // namespace constraint
} // namespace adelie_core