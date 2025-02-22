#pragma once
#include <adelie_core/configs.hpp>
#include <adelie_core/constraint/constraint_one_sided.hpp>
#include <adelie_core/constraint/utils.hpp>
#include <adelie_core/optimization/nnqp_full.hpp>

namespace adelie_core {
namespace constraint {
namespace one_sided {

template <class ValueType>
inline void solve_1d(
    Eigen::Ref<util::rowvec_type<ValueType>> x,
    Eigen::Ref<util::rowvec_type<ValueType>> mu,
    const Eigen::Ref<const util::rowvec_type<ValueType>>& quad,
    const Eigen::Ref<const util::rowvec_type<ValueType>>& linear,
    ValueType l1,
    ValueType l2,
    const Eigen::Ref<const util::colmat_type<ValueType>>& Q,
    const Eigen::Ref<const util::rowvec_type<ValueType>>& _sgn,
    const Eigen::Ref<const util::rowvec_type<ValueType>>& _b
) 
{
    using value_t = ValueType;

    const auto A = _sgn[0] * Q(0,0);
    const auto b = _b[0];
    const auto q = quad[0];
    const auto v = linear[0];

    // check if x == 0 is optimal
    auto mu0 = (b > 0) ? 0 : std::max<value_t>(A * v, 0.0);
    const auto is_zero_opt = std::abs(v - A * mu0) <= l1;

    // if optimal, take previous solution, else compute general solution
    const auto x0 = is_zero_opt ? 0 : (
        A * std::min(
            A * std::copysign(std::abs(v) - l1, v) / (q + l2), 
            b
        )
    );
    mu0 = is_zero_opt ? mu0 : (
        (A * x0 < b) ? 0 : (A * (v - ((q + l2) * x0 + std::copysign(l1, x0))))
    );

    // store output
    x[0] = x0;
    mu[0] = mu0;
}

} // namespace one_sided

ADELIE_CORE_CONSTRAINT_ONE_SIDED_TP
ADELIE_CORE_CONSTRAINT_ONE_SIDED::ConstraintOneSided(
    const Eigen::Ref<const vec_value_t>& sgn,
    const Eigen::Ref<const vec_value_t>& b,
    size_t max_iters,
    value_t tol,
    size_t pinball_max_iters,
    value_t pinball_tol,
    value_t slack
):
    _sgn(sgn.data(), sgn.size()),
    _b(b.data(), b.size()),
    _max_iters(max_iters),
    _tol(tol),
    _pinball_max_iters(pinball_max_iters),
    _pinball_tol(pinball_tol),
    _slack(slack),
    _mu(vec_value_t::Zero(sgn.size()))
{
    if ((_sgn.abs() != 1).any()) {
        throw util::adelie_core_error("sgn must be a vector of +/-1.");
    }
    if ((_b < 0).any()) {
        throw util::adelie_core_error("b must be >= 0.");
    }
    if (_sgn.size() != _b.size()) {
        throw util::adelie_core_error("sgn be (d,) where b is (d,).");
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

ADELIE_CORE_CONSTRAINT_ONE_SIDED_TP
void
ADELIE_CORE_CONSTRAINT_ONE_SIDED::gradient(
    const Eigen::Ref<const vec_value_t>&,
    const Eigen::Ref<const vec_value_t>& mu,
    Eigen::Ref<vec_value_t> out
) const
{
    out = _sgn * mu;
}

ADELIE_CORE_CONSTRAINT_ONE_SIDED_TP
void
ADELIE_CORE_CONSTRAINT_ONE_SIDED::project(
    Eigen::Ref<vec_value_t> x
) const
{
    x = _sgn * (_sgn * x).min(_b);
}

ADELIE_CORE_CONSTRAINT_ONE_SIDED_TP
int
ADELIE_CORE_CONSTRAINT_ONE_SIDED::duals() const 
{ 
    return _b.size(); 
}

ADELIE_CORE_CONSTRAINT_ONE_SIDED_TP
int
ADELIE_CORE_CONSTRAINT_ONE_SIDED::primals() const 
{ 
    return _b.size(); 
}

ADELIE_CORE_CONSTRAINT_ONE_SIDED_TP
size_t
ADELIE_CORE_CONSTRAINT_ONE_SIDED::buffer_size() const 
{
    const auto d = _b.size();
    return (d <= 1) ? 0 : (d * (9 + 2 * d));
}

ADELIE_CORE_CONSTRAINT_ONE_SIDED_TP
void
ADELIE_CORE_CONSTRAINT_ONE_SIDED::solve(
    Eigen::Ref<vec_value_t> x,
    const Eigen::Ref<const vec_value_t>& quad,
    const Eigen::Ref<const vec_value_t>& linear,
    value_t l1,
    value_t l2,
    const Eigen::Ref<const colmat_value_t>& Q,
    Eigen::Ref<vec_uint64_t> buffer
) 
{
    const auto m = _b.size();
    const auto d = m;

    if (d == 1) {
        one_sided::solve_1d<value_t>(x, _mu, quad, linear, l1, l2, Q, _sgn, _b);
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
        mu_resid.matrix() = linear.matrix() - (_sgn * _mu).matrix() * Q;
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
        const auto is_b_zero = (_b <= 0).template cast<value_t>();
        _mu = (_sgn * Qv).max(0).min(
            Configs::max_solver_value * is_b_zero
        );
        const auto mu_resid_norm_sq = (Qv - _sgn * _mu).square().sum();
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
        return ((_sgn * Qv - _mu) * (_mu - mu_prev)).sum();
    };
    const auto compute_backtrack = [&](auto step_size) {
        _mu = mu_prev + step_size * (_mu - mu_prev);
    };
    const auto compute_gradient = [&]() {
        grad.matrix() = (x.matrix() * Q.transpose()).cwiseProduct(_sgn.matrix()) - _b.matrix();
    };
    const auto compute_hard_optimality = [&]() {
        return (grad <= 0).all() && (_mu * grad == 0).all();
    };
    const auto compute_convergence_measure = [&](
        bool is_in_ellipse
    ) {
        return is_in_ellipse ? (
            std::abs(((_mu-mu_prev) * (grad_prev+_b)).mean())
        ) : (
            std::abs(((_mu-mu_prev) * (grad_prev-grad)).mean())
        );
    };
    const auto compute_proximal_newton_step = [&](
        const auto& hess,
        const auto var
    ) {
        // reparametrize
        grad *= _sgn;
        _mu *= _sgn;

        // solve NNQP for new mu
        optimization::StateNNQPFull<colmat_value_t, true> state_nnqp(
            _sgn, hess, var, _pinball_max_iters, _pinball_tol, _mu, grad
        );
        state_nnqp.solve();

        // reparametrize
        _mu *= _sgn;
    };
    const auto save_additional_prev = [&](bool is_in_ellipse) {
        mu_prev = _mu;
        if (is_in_ellipse) grad_prev = -_b;
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

ADELIE_CORE_CONSTRAINT_ONE_SIDED_TP
void
ADELIE_CORE_CONSTRAINT_ONE_SIDED::gradient(
    const Eigen::Ref<const vec_value_t>&,
    Eigen::Ref<vec_value_t> out
) const 
{
    out = _sgn * _mu;
}

ADELIE_CORE_CONSTRAINT_ONE_SIDED_TP
typename ADELIE_CORE_CONSTRAINT_ONE_SIDED::value_t
ADELIE_CORE_CONSTRAINT_ONE_SIDED::solve_zero(
    const Eigen::Ref<const vec_value_t>& v,
    Eigen::Ref<vec_uint64_t> 
) 
{
    const auto is_b_zero = (_b <= 0).template cast<value_t>();
    _mu = (_sgn * v).max(0).min(
        Configs::max_solver_value * is_b_zero
    );
    return (v - _sgn * _mu).matrix().norm();
};

ADELIE_CORE_CONSTRAINT_ONE_SIDED_TP
void
ADELIE_CORE_CONSTRAINT_ONE_SIDED::clear() 
{
    _mu.setZero();
}

ADELIE_CORE_CONSTRAINT_ONE_SIDED_TP
void
ADELIE_CORE_CONSTRAINT_ONE_SIDED::dual(
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

ADELIE_CORE_CONSTRAINT_ONE_SIDED_TP
int
ADELIE_CORE_CONSTRAINT_ONE_SIDED::duals_nnz() const 
{
    return (_mu != 0).count();
}

ADELIE_CORE_CONSTRAINT_ONE_SIDED_ADMM_TP
ADELIE_CORE_CONSTRAINT_ONE_SIDED_ADMM::ConstraintOneSidedADMM(
    const Eigen::Ref<const vec_value_t>& sgn,
    const Eigen::Ref<const vec_value_t>& b,
    size_t max_iters,
    value_t tol_abs,
    value_t tol_rel,
    value_t rho
):
    _sgn(sgn.data(), sgn.size()),
    _b(b.data(), b.size()),
    _max_iters(max_iters),
    _tol_abs(tol_abs),
    _tol_rel(tol_rel),
    _rho(rho),
    _mu(vec_value_t::Zero(sgn.size()))
{
    if (tol_abs < 0) {
        throw util::adelie_core_error("tol_abs must be >= 0.");
    }
    if (tol_rel < 0) {
        throw util::adelie_core_error("tol_rel must be >= 0.");
    }
    if (rho <= 0) {
        throw util::adelie_core_error("rho must be > 0.");
    }
}

ADELIE_CORE_CONSTRAINT_ONE_SIDED_ADMM_TP
void
ADELIE_CORE_CONSTRAINT_ONE_SIDED_ADMM::gradient(
    const Eigen::Ref<const vec_value_t>&,
    const Eigen::Ref<const vec_value_t>& mu,
    Eigen::Ref<vec_value_t> out
) const
{
    out = _sgn * mu;
}

ADELIE_CORE_CONSTRAINT_ONE_SIDED_ADMM_TP
void
ADELIE_CORE_CONSTRAINT_ONE_SIDED_ADMM::project(
    Eigen::Ref<vec_value_t> x
) const
{
    x = _sgn * (_sgn * x).min(_b);
}

ADELIE_CORE_CONSTRAINT_ONE_SIDED_ADMM_TP
int
ADELIE_CORE_CONSTRAINT_ONE_SIDED_ADMM::duals() const 
{ 
    return _b.size(); 
}

ADELIE_CORE_CONSTRAINT_ONE_SIDED_ADMM_TP
int
ADELIE_CORE_CONSTRAINT_ONE_SIDED_ADMM::primals() const 
{ 
    return _b.size(); 
}

ADELIE_CORE_CONSTRAINT_ONE_SIDED_ADMM_TP
size_t
ADELIE_CORE_CONSTRAINT_ONE_SIDED_ADMM::buffer_size() const 
{
    const auto d = _b.size();
    return (d <= 1) ? 0 : (7 * d);
}

ADELIE_CORE_CONSTRAINT_ONE_SIDED_ADMM_TP
void
ADELIE_CORE_CONSTRAINT_ONE_SIDED_ADMM::solve(
    Eigen::Ref<vec_value_t> x,
    const Eigen::Ref<const vec_value_t>& quad,
    const Eigen::Ref<const vec_value_t>& linear,
    value_t l1,
    value_t l2,
    const Eigen::Ref<const colmat_value_t>& Q,
    Eigen::Ref<vec_uint64_t> buffer
) 
{
    const auto m = _b.size();
    const auto d = m;

    if (d == 1) {
        one_sided::solve_1d<value_t>(x, _mu, quad, linear, l1, l2, Q, _sgn, _b);
        return;
    } 

    const auto& v = linear;

    size_t iters = 0;

    auto buff_ptr = reinterpret_cast<value_t*>(buffer.data());
    Eigen::Map<vec_value_t> x_buffer1(buff_ptr, d); buff_ptr += d;
    Eigen::Map<vec_value_t> x_buffer2(buff_ptr, d); buff_ptr += d;
    Eigen::Map<vec_value_t> z(buff_ptr, m); buff_ptr += m;
    Eigen::Map<vec_value_t> z_prev(buff_ptr, m); buff_ptr += m;
    Eigen::Map<vec_value_t> u(buff_ptr, m); buff_ptr += m;
    Eigen::Map<vec_value_t> linear_shifted(buff_ptr, d); buff_ptr += d;
    Eigen::Map<vec_value_t> s(buff_ptr, m); buff_ptr += m;

    z = ((x.matrix() * Q.transpose()).array() * _sgn).min(_b);
    u.setZero();

    const auto compute_primal_x = [&]() {
        constexpr size_t _newton_max_iters = 100000;
        constexpr value_t _newton_tol = 1e-12;
        size_t x_iters;
        bcd::unconstrained::newton_solver(
            quad, linear_shifted, l1, l2+_rho, _newton_tol, _newton_max_iters, 
            x, x_iters, x_buffer1, x_buffer2
        );
    };

    const auto compute_primal_z = [&]() {
        z = linear_shifted.min(_b);
    };

    const auto compute_dual = [&]() {
        const auto x_norm = x.matrix().norm();
        if (x_norm <= 0) {
            _mu = (
                ((v.matrix() * Q.transpose()).array() * _sgn).max(0) 
                * (_b <= 0).template cast<value_t>()
            );
            return;
        }
        _mu = (((v - (quad + l2 + l1 / x_norm) * x).matrix() * Q.transpose()).array() * _sgn).max(0);
    };

    while (iters < _max_iters) {
        ++iters;
        linear_shifted.matrix() = v.matrix() + _rho * (_sgn * (z - u)).matrix() * Q;
        compute_primal_x();
        linear_shifted = (x.matrix() * Q.transpose()).array() * _sgn + u;
        z_prev = z;
        compute_primal_z();
        auto& r = linear_shifted;
        r = linear_shifted - z - u;
        u += r;
        s.matrix() = -_rho * (_sgn * (z-z_prev)).matrix() * Q;
        const value_t eps_pri = (
            std::sqrt(m) * _tol_abs
            + _tol_rel * std::max(x.matrix().norm(), z.matrix().norm())
        );
        const value_t eps_dual = (
            std::sqrt(d) * _tol_abs 
            + _tol_rel * _rho * u.matrix().norm()
        );
        if (
            (r.matrix().norm() <= eps_pri) &&
            (s.matrix().norm() <= eps_dual)
        ) {
            compute_dual();
            return;
        }
    }

    throw util::adelie_core_solver_error("ConstraintOneSidedADMM: max iterations reached!");
}

ADELIE_CORE_CONSTRAINT_ONE_SIDED_ADMM_TP
void
ADELIE_CORE_CONSTRAINT_ONE_SIDED_ADMM::gradient(
    const Eigen::Ref<const vec_value_t>&,
    Eigen::Ref<vec_value_t> out
) const
{
    out = _sgn * _mu;
}

ADELIE_CORE_CONSTRAINT_ONE_SIDED_ADMM_TP
typename ADELIE_CORE_CONSTRAINT_ONE_SIDED_ADMM::value_t
ADELIE_CORE_CONSTRAINT_ONE_SIDED_ADMM::solve_zero(
    const Eigen::Ref<const vec_value_t>& v,
    Eigen::Ref<vec_uint64_t> 
) 
{
    const auto is_b_zero = (_b <= 0).template cast<value_t>();
    _mu = (_sgn * v).max(0).min(
        Configs::max_solver_value * is_b_zero
    );
    return (v - _sgn * _mu).matrix().norm();
};

ADELIE_CORE_CONSTRAINT_ONE_SIDED_ADMM_TP
void
ADELIE_CORE_CONSTRAINT_ONE_SIDED_ADMM::clear() 
{
    _mu.setZero();
}

ADELIE_CORE_CONSTRAINT_ONE_SIDED_ADMM_TP
void
ADELIE_CORE_CONSTRAINT_ONE_SIDED_ADMM::dual(
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

ADELIE_CORE_CONSTRAINT_ONE_SIDED_ADMM_TP
int
ADELIE_CORE_CONSTRAINT_ONE_SIDED_ADMM::duals_nnz() const 
{
    return (_mu != 0).count();
}

} // namespace constraint
} // namespace adelie_core