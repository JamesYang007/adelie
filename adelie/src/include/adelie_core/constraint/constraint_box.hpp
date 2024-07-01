#pragma once
#include <adelie_core/constraint/constraint_base.hpp>
#include <adelie_core/optimization/hinge_full.hpp>

namespace adelie_core {
namespace constraint {

template <class ValueType>
class ConstraintBoxBase: public ConstraintBase<ValueType>
{
public:
    using base_t = ConstraintBase<ValueType>;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_uint64_t;
    using typename base_t::colmat_value_t;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;

protected:
    const map_cvec_value_t _l;
    const map_cvec_value_t _u;

public:
    explicit ConstraintBoxBase(
        const Eigen::Ref<const vec_value_t> l,
        const Eigen::Ref<const vec_value_t> u
    ):
        _l(l.data(), l.size()),
        _u(u.data(), u.size())
    {
        if (_u.size() != _l.size()) {
            throw util::adelie_core_error("upper and lower must have the same length.");
        }
        if ((_u < 0).any()) {
            throw util::adelie_core_error("upper must be >= 0.");
        }
        // NOTE: user passes in lower == -l
        if ((_l < 0).any()) { 
            throw util::adelie_core_error("lower must be <= 0.");
        }
    }

    ADELIE_CORE_STRONG_INLINE
    void solve_1d(
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

    void project(
        Eigen::Ref<vec_value_t> x
    ) override
    {
        x = (x).min(_u).max(-_l);
    }

    void gradient(
        const Eigen::Ref<const vec_value_t>&,
        const Eigen::Ref<const vec_value_t>& mu,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        out = mu;
    }

    int duals() override { return _u.size(); }
    int primals() override { return _u.size(); }
};

template <class ValueType>
class ConstraintBoxProximalNewton: public ConstraintBoxBase<ValueType>
{
public:
    using base_t = ConstraintBoxBase<ValueType>;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_uint64_t;
    using typename base_t::colmat_value_t;
    using typename base_t::map_cvec_value_t;
    using base_t::_l;
    using base_t::_u;

private:
    const size_t _max_iters;
    const value_t _tol;
    const size_t _nnls_max_iters;
    const value_t _nnls_tol;
    const value_t _cs_tol;
    const value_t _slack;

public:
    explicit ConstraintBoxProximalNewton(
        const Eigen::Ref<const vec_value_t> l,
        const Eigen::Ref<const vec_value_t> u,
        size_t max_iters,
        value_t tol,
        size_t nnls_max_iters,
        value_t nnls_tol,
        value_t cs_tol,
        value_t slack
    ):
        base_t(l, u),
        _max_iters(max_iters),
        _tol(tol),
        _nnls_max_iters(nnls_max_iters),
        _nnls_tol(nnls_tol),
        _cs_tol(cs_tol),
        _slack(slack)
    {
        if (tol < 0) {
            throw util::adelie_core_error("tol must be >= 0.");
        }
        if (nnls_tol < 0) {
            throw util::adelie_core_error("nnls_tol must be >= 0.");
        }
        if (cs_tol < 0) {
            throw util::adelie_core_error("cs_tol must be >= 0.");
        }
        if (slack <= 0 || slack >= 1) {
            throw util::adelie_core_error("slack must be in (0,1).");
        }
    }

    size_t buffer_size() override 
    {
        const auto d = _l.size();
        return (d <= 1) ? 0 : (d * (9 + 2 * d));
    }

    void solve(
        Eigen::Ref<vec_value_t> x,
        Eigen::Ref<vec_value_t> mu,
        const Eigen::Ref<const vec_value_t>& quad,
        const Eigen::Ref<const vec_value_t>& linear,
        value_t l1,
        value_t l2,
        const Eigen::Ref<const colmat_value_t>& Q,
        Eigen::Ref<vec_uint64_t> buffer
    ) override
    {
        const auto m = _u.size();
        const auto d = m;

        if (d == 1) {
            base_t::solve_1d(x, mu, quad, linear, l1, l2, Q);
            return;
        }

        auto buff_ptr = reinterpret_cast<value_t*>(buffer.data());
        const auto buff_begin = buff_ptr;
        Eigen::Map<vec_value_t> grad_prev(buff_ptr, m); buff_ptr += m;
        Eigen::Map<vec_value_t> grad(buff_ptr, m); buff_ptr += m;
        const auto n_read = std::distance(buff_begin, buff_ptr);
        Eigen::Map<vec_uint64_t> next_buff(buffer.data() + n_read, buffer.size() - n_read);

        const auto compute_mu_resid = [&](
            const auto& mu,
            const auto& linear,
            const auto& Q,
            auto& mu_resid
        ) {
            mu_resid.matrix() = linear.matrix() - mu.matrix() * Q;
        };
        const auto compute_hard_min_mu_resid = [&](
            auto& mu,
            const auto& Qv
        ) {
            mu = Qv;
            return 0;
        };
        const auto compute_soft_min_mu_resid = [&](
            auto& mu,
            const auto& Qv
        ) {
            mu = (
                mu.max(0) * (_u <= 0).template cast<value_t>()
                + mu.min(0) * (_l <= 0).template cast<value_t>()
            );
            return (Qv - mu).square().sum();
        };
        const auto compute_relaxed_slackness = [&](
            const auto& mu
        ) {
            return std::max(
                (mu.max(0) * _u).square().mean(),
                (mu.min(0) * _l).square().mean()
            );
        };
        const auto compute_backtrack_a = [&](
            const auto& mu_prev,
            const auto& mu
        ) {
            return (mu - mu_prev).square().sum();
        };
        const auto compute_backtrack_b = [&](
            const auto& mu_prev,
            const auto& mu,
            const auto& Qv,
            const auto& 
        ) {
            return ((Qv - mu) * (mu - mu_prev)).sum();
        };
        const auto compute_gradient = [&](
            const auto& x,
            const auto& Q
        ) {
            grad.matrix() = x.matrix() * Q.transpose();
        };
        const auto compute_hard_optimality = [&](
            const auto& mu
        ) {
            return (
                ((grad <= _u) && (grad >= -_l)).all() && 
                (mu.max(0) * (grad - _u) == 0).all() &&
                (mu.min(0) * (grad + _l) == 0).all()
            );
        };
        const auto compute_convergence_measure = [&](
            const auto& mu_prev,
            const auto& mu,
            bool is_in_ellipse
        ) {
            return is_in_ellipse ? (
                std::abs(((mu-mu_prev) * grad_prev).mean())
            ) : (
                std::abs(((mu-mu_prev) * (grad_prev-grad)).mean())
            );
        };
        const auto compute_proximal_newton_step = [&](
            const auto& hess,
            const auto x_norm,
            auto& mu
        ) {
            optimization::StateHingeFull<colmat_value_t> state_hinge(
                hess, _u, _l, _nnls_max_iters, _nnls_tol * std::max<value_t>(x_norm, 1), mu, grad 
            );
            optimization::hinge_full(state_hinge);
        };
        const auto save_additional_prev = [&](bool is_in_ellipse) {
            if (is_in_ellipse) grad_prev.setZero();
            else grad_prev = grad;
        };
        base_t::_solve_proximal_newton(
            x, mu, quad, linear, l1, l2, Q, _max_iters, _tol, _cs_tol, _slack, next_buff,
            compute_mu_resid,
            compute_hard_min_mu_resid,
            compute_soft_min_mu_resid,
            compute_relaxed_slackness,
            compute_backtrack_a,
            compute_backtrack_b,
            compute_gradient,
            compute_hard_optimality,
            compute_convergence_measure,
            compute_proximal_newton_step,
            save_additional_prev
        );
    }
};

} // namespace constraint
} // namespace adelie_core