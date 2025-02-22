#pragma once
#include <adelie_core/configs.hpp>
#include <adelie_core/constraint/constraint_box.hpp>
#include <adelie_core/constraint/utils.hpp>
#include <adelie_core/optimization/pinball_full.hpp>

namespace adelie_core {
namespace constraint {

ADELIE_CORE_CONSTRAINT_BOX_TP
ADELIE_CORE_CONSTRAINT_BOX::ConstraintBox(
    const Eigen::Ref<const vec_value_t>& l,
    const Eigen::Ref<const vec_value_t>& u,
    size_t max_iters,
    value_t tol,
    size_t pinball_max_iters,
    value_t pinball_tol,
    value_t slack
):
    _l(l.data(), l.size()),
    _u(u.data(), u.size()),
    _max_iters(max_iters),
    _tol(tol),
    _pinball_max_iters(pinball_max_iters),
    _pinball_tol(pinball_tol),
    _slack(slack),
    _mu(vec_value_t::Zero(l.size()))
{
    const auto d = _u.size();
    if (_l.size() != d) {
        throw util::adelie_core_error("lower must be (d,) where upper is (d,).");
    }
    if ((_u < 0).any()) {
        throw util::adelie_core_error("upper must be >= 0.");
    }
    if ((_l < 0).any()) { 
        // NOTE: user passes in lower == -l
        throw util::adelie_core_error("lower must be <= 0.");
    }
    if (tol < 0) {
        throw util::adelie_core_error("tol must be >= 0.");
    }
    if (pinball_tol < 0) {
        throw util::adelie_core_error("pinball_tol must be >= 0.");
    }
    if (slack <= 0 || slack >= 1) {
        throw util::adelie_core_error("slack must be in (0,1).");
    }
}

ADELIE_CORE_CONSTRAINT_BOX_TP
void 
ADELIE_CORE_CONSTRAINT_BOX::solve_1d(
    Eigen::Ref<vec_value_t> x,
    Eigen::Ref<vec_value_t> mu,
    const Eigen::Ref<const vec_value_t>& quad,
    const Eigen::Ref<const vec_value_t>& linear,
    value_t l1,
    value_t l2,
    const Eigen::Ref<const colmat_value_t>& Q
) const
{
    const auto A = Q(0,0);
    const auto u = _u[0];
    const auto l = _l[0];
    const auto q = quad[0];
    const auto v = linear[0];

    // check if x == 0 is optimal
    auto mu0_pos = (u > 0) ? 0 : std::max<value_t>(A * v, 0.0);
    auto mu0_neg = (l > 0) ? 0 : std::max<value_t>(-A * v, 0.0);
    auto mu0 = mu0_pos - mu0_neg;
    const auto is_zero_opt = std::abs(v - A * mu0) <= l1;

    // if optimal, take previous solution, else compute general solution
    const auto x0 = is_zero_opt ? 0 : (
        A * std::max(std::min(
            A * std::copysign(std::abs(v) - l1, v) / (q + l2), 
            u 
        ), -l)
    );
    const auto mu0_full = A * (v - ((q + l2) * x0 + std::copysign(l1, x0)));
    mu0_pos = is_zero_opt ? mu0_pos : (
        (A * x0 < u) ? 0 : std::max<value_t>(mu0_full, 0)
    );
    mu0_neg = is_zero_opt ? mu0_neg : (
        (A * x0 > -l) ? 0 : std::max<value_t>(-mu0_full, 0)
    );
    mu0 = mu0_pos - mu0_neg;

    // store output
    x[0] = x0;
    mu[0] = mu0;
}

ADELIE_CORE_CONSTRAINT_BOX_TP
void 
ADELIE_CORE_CONSTRAINT_BOX::project(
    Eigen::Ref<vec_value_t> x
) const
{
    x = x.min(_u).max(-_l);
}

ADELIE_CORE_CONSTRAINT_BOX_TP
int 
ADELIE_CORE_CONSTRAINT_BOX::duals() const 
{ 
    return _u.size(); 
}

ADELIE_CORE_CONSTRAINT_BOX_TP
int 
ADELIE_CORE_CONSTRAINT_BOX::primals() const 
{ 
    return _u.size(); 
}

ADELIE_CORE_CONSTRAINT_BOX_TP
size_t 
ADELIE_CORE_CONSTRAINT_BOX::buffer_size() const 
{
    const auto d = _l.size();
    return (d <= 1) ? 0 : (d * (9 + 2 * d));
}

ADELIE_CORE_CONSTRAINT_BOX_TP
void 
ADELIE_CORE_CONSTRAINT_BOX::solve(
    Eigen::Ref<vec_value_t> x,
    const Eigen::Ref<const vec_value_t>& quad,
    const Eigen::Ref<const vec_value_t>& linear,
    value_t l1,
    value_t l2,
    const Eigen::Ref<const colmat_value_t>& Q,
    Eigen::Ref<vec_uint64_t> buffer
) 
{
    const auto m = _u.size();
    const auto d = m;

    if (d == 1) {
        solve_1d(x, _mu, quad, linear, l1, l2, Q);
        return;
    }

    // check if x = 0, mu = 0 is optimal
    if (linear.matrix().norm() <= l1) {
        x.setZero();
        _mu.setZero();
        return;
    }

    auto buff_ptr = reinterpret_cast<value_t*>(buffer.data());
    const auto buff_begin = buff_ptr;
    Eigen::Map<vec_value_t> grad_prev(buff_ptr, m); buff_ptr += m;
    Eigen::Map<vec_value_t> grad(buff_ptr, m); buff_ptr += m;
    Eigen::Map<vec_value_t> mu_prev(buff_ptr, m); buff_ptr += m;
    const auto n_read = std::distance(buff_begin, buff_ptr);
    Eigen::Map<vec_uint64_t> next_buff(buffer.data() + n_read, buffer.size() - n_read);

    const auto compute_mu_resid = [&](
        auto& mu_resid
    ) {
        mu_resid.matrix() = linear.matrix() - _mu.matrix() * Q;
    };
    const auto compute_min_mu_resid = [&](
        const auto& Qv,
        bool is_prev_valid_old,
        bool is_init
    ) {
        auto& mu_curr = grad;
        if (is_prev_valid_old || is_init) {
            mu_curr = _mu;
        }
        const auto is_u_zero = (_u <= 0).template cast<value_t>();
        const auto is_l_zero = (_l <= 0).template cast<value_t>();
        _mu = Qv.max(
            (-Configs::max_solver_value) * is_l_zero
        ).min(
            Configs::max_solver_value * is_u_zero
        );
        const auto mu_resid_norm_sq = (Qv - _mu).square().sum();
        if ((is_init || is_prev_valid_old) && mu_resid_norm_sq > l1 * l1) {
            _mu = mu_curr;
        }
        return mu_resid_norm_sq;
    };
    const auto compute_backtrack_a = [&]() {
        return (_mu - mu_prev).square().sum();
    };
    const auto compute_backtrack_b = [&](
        const auto& Qv,
        const auto& 
    ) {
        return ((Qv - _mu) * (_mu - mu_prev)).sum();
    };
    const auto compute_backtrack = [&](auto step_size) {
        _mu = mu_prev + step_size * (_mu - mu_prev);
    };
    const auto compute_gradient = [&]() {
        grad.matrix() = x.matrix() * Q.transpose();
    };
    const auto compute_hard_optimality = [&]() {
        return (
            ((grad <= _u) && (grad >= -_l)).all() && 
            (_mu.max(0) * (grad - _u) == 0).all() &&
            (_mu.min(0) * (grad + _l) == 0).all()
        );
    };
    const auto compute_convergence_measure = [&](
        bool is_in_ellipse
    ) {
        return is_in_ellipse ? (
            std::abs(((_mu-mu_prev) * grad_prev).mean())
        ) : (
            std::abs(((_mu-mu_prev) * (grad_prev-grad)).mean())
        );
    };
    const auto compute_proximal_newton_step = [&](
        const auto& hess,
        const auto var
    ) {
        optimization::StatePinballFull<colmat_value_t> state_pinball(
            hess, _l, _u, var, _pinball_max_iters, _pinball_tol, _mu, grad 
        );
        state_pinball.solve();
    };
    const auto save_additional_prev = [&](bool is_in_ellipse) {
        mu_prev = _mu;
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

ADELIE_CORE_CONSTRAINT_BOX_TP
void 
ADELIE_CORE_CONSTRAINT_BOX::gradient(
    const Eigen::Ref<const vec_value_t>&,
    Eigen::Ref<vec_value_t> out
) const
{
    out = _mu;
}

ADELIE_CORE_CONSTRAINT_BOX_TP
void 
ADELIE_CORE_CONSTRAINT_BOX::gradient(
    const Eigen::Ref<const vec_value_t>&,
    const Eigen::Ref<const vec_value_t>& mu,
    Eigen::Ref<vec_value_t> out
) const 
{
    out = mu;
}

ADELIE_CORE_CONSTRAINT_BOX_TP
typename ADELIE_CORE_CONSTRAINT_BOX::value_t
ADELIE_CORE_CONSTRAINT_BOX::solve_zero(
    const Eigen::Ref<const vec_value_t>& v,
    Eigen::Ref<vec_uint64_t> 
) 
{
    const auto is_u_zero = (_u <= 0).template cast<value_t>();
    const auto is_l_zero = (_l <= 0).template cast<value_t>();
    _mu = v.max(
        (-Configs::max_solver_value) * is_l_zero
    ).min(
        Configs::max_solver_value * is_u_zero
    );
    return (v - _mu).matrix().norm();
};

ADELIE_CORE_CONSTRAINT_BOX_TP
void
ADELIE_CORE_CONSTRAINT_BOX::clear() 
{
    _mu.setZero();
}

ADELIE_CORE_CONSTRAINT_BOX_TP
void 
ADELIE_CORE_CONSTRAINT_BOX::dual(
    Eigen::Ref<vec_index_t> indices,
    Eigen::Ref<vec_value_t> values
) const
{
    size_t nnz = 0;
    for (Eigen::Index i = 0; i < _mu.size(); ++i) {
        const auto mi = _mu[i];
        if (mi == 0) continue;
        indices[nnz] = i;
        values[nnz] = mi;
        ++nnz;
    }
}

ADELIE_CORE_CONSTRAINT_BOX_TP
int 
ADELIE_CORE_CONSTRAINT_BOX::duals_nnz() const 
{
    return (_mu != 0).count();
}

} // namespace constraint
} // namespace adelie_core