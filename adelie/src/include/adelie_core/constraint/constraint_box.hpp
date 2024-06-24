#pragma once
#include <adelie_core/constraint/constraint_base.hpp>
#include <adelie_core/bcd/unconstrained/newton.hpp>
#include <adelie_core/optimization/lasso_full.hpp>
#include <adelie_core/util/macros.hpp>

namespace adelie_core {
namespace constraint {

template <class ValueType>
class ConstraintBoxBase: public ConstraintBase<ValueType>
{
public:
    using base_t = ConstraintBase<ValueType>;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
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
            throw util::adelie_core_error("u and l must have the same length.");
        }
        if ((_u < 0).any()) {
            throw util::adelie_core_error("u must be >= 0.");
        }
        if ((_l < 0).any()) {
            throw util::adelie_core_error("l must be >= 0.");
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
    using typename base_t::colmat_value_t;
    using typename base_t::map_cvec_value_t;
    using base_t::_l;
    using base_t::_u;

private:
    const size_t _max_iters;
    const value_t _tol;
    const size_t _newton_max_iters = 100000;
    const value_t _newton_tol = 1e-12;
    const size_t _nnls_max_iters;
    const value_t _nnls_tol;
    const value_t _cs_tol;
    const value_t _slack;
    const vec_value_t _penalty;
    vec_value_t _buff;

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
        _slack(slack),
        _penalty(0.5 * (l + u)),
        _buff((l.size() <= 1) ? 0 : (l.size() * (9 + 2 * l.size())))
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

    void solve(
        Eigen::Ref<vec_value_t> x,
        Eigen::Ref<vec_value_t> mu,
        const Eigen::Ref<const vec_value_t>& quad,
        const Eigen::Ref<const vec_value_t>& linear,
        value_t l1,
        value_t l2,
        const Eigen::Ref<const colmat_value_t>& Q
    ) override
    {
        using rowmat_value_t = util::rowmat_type<value_t>;

        const auto m = _u.size();
        const auto d = m;

        if (d == 1) {
            base_t::solve_1d(x, mu, quad, linear, l1, l2, Q);
            return;
        }

        const auto& v = linear;

        size_t iters = 0;

        auto buff_ptr = _buff.data();
        Eigen::Map<vec_value_t> x_buffer1(buff_ptr, d); buff_ptr += d;
        Eigen::Map<vec_value_t> x_buffer2(buff_ptr, d); buff_ptr += d;
        Eigen::Map<vec_value_t> mu_resid(buff_ptr, d); buff_ptr += d;
        Eigen::Map<vec_value_t> mu_prev(buff_ptr, m); buff_ptr += m;
        Eigen::Map<vec_value_t> grad_prev(buff_ptr, m); buff_ptr += m;
        Eigen::Map<vec_value_t> grad(buff_ptr, m); buff_ptr += m;
        Eigen::Map<rowmat_value_t> hess_buff(buff_ptr, m, d); buff_ptr += m * d;
        Eigen::Map<colmat_value_t> hess(buff_ptr, m, m); buff_ptr += m * m;
        Eigen::Map<vec_value_t> alpha_tmp(buff_ptr, d); buff_ptr += d;
        Eigen::Map<vec_value_t> alpha(buff_ptr, m); buff_ptr += m;
        Eigen::Map<vec_value_t> Qv(buff_ptr, d); buff_ptr += d;

        bool is_prev_valid = false;
        bool zero_primal_checked = false;
        value_t mu_resid_norm_prev = -1;

        const auto compute_mu_resid = [&]() {
            mu_resid.matrix() = v.matrix() - mu.matrix() * Q;
        };
        const auto compute_primal = [&]() {
            size_t x_iters;
            bcd::unconstrained::newton_abs_solver(
                quad, mu_resid, l1, l2, _newton_tol, _newton_max_iters, 
                x, x_iters, x_buffer1, x_buffer2
            );
        };

        while (iters < _max_iters) {
            ++iters;

            compute_mu_resid();
            const value_t mu_resid_norm = mu_resid.matrix().norm();
            const value_t mu_resid_norm_sq = mu_resid_norm * mu_resid_norm;

            // Only used if first is_in_ellipse check fails.
            value_t x_norm = -1;

            // Check if x^star(mu) = 0 (i.e. in the ellipse).
            // The first check is a quick way to check the condition.
            // Sometimes, is_in_ellipse may be `true` but the compute_primal() may still return x=0.
            // We do a second check if the first condition fails to be consistent with the primal.
            bool is_in_ellipse = mu_resid_norm <= l1;
            if (!is_in_ellipse) {
                compute_primal();
                x_norm = x.matrix().norm();
                is_in_ellipse = x_norm <= 0;
            }

            // Check if x^star(mu) == 0.
            if (is_in_ellipse) {
                // NOTE: this check is important since numerical precision issues
                // may make us enter this loop infinitely.
                if (is_prev_valid) {
                    const auto convg_meas = std::abs(
                        ((mu-mu_prev) * grad_prev).sum()
                    ) / m;
                    if (convg_meas <= _tol) {
                        x.setZero();
                        return;
                    }
                }

                // Check if there is a primal-dual optimal pair where primal = 0.
                // To be optimal, they must satisfy the 4 KKT conditions.
                // NOTE: the if-block below is only entered at most once. After it has been entered once,
                // it guarantees that after the next iteration is_prev_valid == true.
                // So subsequent access to the current block guarantees backtracking.
                if (!zero_primal_checked) {
                    zero_primal_checked = true;

                    // one-time population
                    Qv.matrix() = v.matrix() * Q.transpose();
                    
                    // If previous is valid, we will use mu_prev and mu_curr to backtrack.
                    // Otherwise, mu is the next iterate.
                    auto& mu_curr = alpha;
                    const bool is_prev_valid_old = is_prev_valid;
                    if (is_prev_valid_old) {
                        mu_curr = mu; // optimization
                    } else {
                        mu_resid_norm_prev = mu_resid_norm;
                        mu_prev = mu;
                        grad_prev.setZero();
                        is_prev_valid = true;
                    }

                    if (
                        ((Qv * _u).square().mean() <= _cs_tol) &&
                        ((Qv * _l).square().mean() <= _cs_tol)
                     ) {
                        x.setZero();
                        return;
                    }

                    mu = (
                        Qv.max(0) * (_u <= 0).template cast<value_t>()
                        + Qv.min(0) * (_l <= 0).template cast<value_t>()
                    );
                    value_t nnls_loss = (Qv - mu).square().sum();
                    if (nnls_loss <= l1 * l1) {
                        x.setZero();
                        return;
                    }

                    if (is_prev_valid_old) mu = mu_curr; // optimization
                    else continue;
                }

                // If we ever enter this region of code, it means that
                // there is no primal-dual optimal pair where primal = 0.
                // This must mean that if there is no previous mu,
                // we must have come from the if-block above, in which case nnls_loss > l1 * l1
                // so the next iteration with the current mu will give a non-zero primal (start proximal Newton from here);
                // otherwise, the proximal newton step overshot so we must backtrack.
                if (!is_prev_valid || (mu_resid_norm_prev <= l1) || (mu_resid_norm_sq > l1 * l1)) {
                    throw util::adelie_core_error(
                        "Possibly an unexpected error! "
                        "Previous iterate should have been properly initialized. "
                        "This may occur if cs_tol is too small. "
                        "If increasing cs_tol does not fix the issue, "
                        "please report this as a bug! "
                    );
                }
                const value_t lmda_target = (1-_slack) * l1 + _slack * mu_resid_norm_prev;
                const value_t a = (mu - mu_prev).square().sum();
                const value_t b = ((Qv - mu) * (mu - mu_prev)).sum();
                const value_t c = mu_resid_norm_sq - lmda_target * lmda_target;
                const value_t t_star = (-b + std::sqrt(std::max<value_t>(b * b - a * c, 0.0))) / a;
                const value_t step_size = std::min<value_t>(std::max<value_t>(1-t_star, 0.0), 1.0);
                mu = mu_prev + step_size * (mu - mu_prev);
                continue;
            }

            grad.matrix() = x.matrix() * Q.transpose();

            // optimization: if optimality hard-check is satisfied, finish early.
            if (
                ((grad <= _u) && (grad >= -_l)).all() && 
                (mu.max(0) * (grad - _u) == 0).all() &&
                (mu.min(0) * (grad + _l) == 0).all()
            ) return;

            // Check if mu is not changing much w.r.t. hessian scaling.
            if (is_prev_valid) {
                const auto convg_meas = std::abs(
                    ((mu-mu_prev) * (grad_prev-grad)).sum()
                ) / m;
                if (convg_meas <= _tol) return;
            }

            // save old values
            mu_resid_norm_prev = mu_resid_norm;
            mu_prev = mu;
            grad_prev = grad;
            is_prev_valid = true;

            // adjust for lasso solver
            grad -= 0.5 * (_u - _l);

            // Compute hessian
            // NOTE:
            //  - x_buffer1 = quad + l2
            //  - x_buffer2 = 1 / (x_buffer1 * x_norm + lmda)

            hess.setZero();
            auto hess_lower = hess.template selfadjointView<Eigen::Lower>();

            // lower(hess) += x_norm * Q diag(x_buffer2) Q^T
            hess_buff.array() = Q.array().rowwise() * x_buffer2.sqrt();
            hess_lower.rankUpdate(hess_buff, x_norm);

            // lower(hess) += x_norm * lmda * kappa * alpha alpha^T
            alpha_tmp = (x * x_buffer2) / x_norm;
            alpha.matrix() = alpha_tmp.matrix() * Q.transpose();
            const auto l1_kappa_norm = l1 * x_norm / (x * x_buffer1 * alpha_tmp).sum();
            hess_lower.rankUpdate(alpha.matrix().transpose(), l1_kappa_norm);

            // full hessian update
            hess.template triangularView<Eigen::Upper>() = hess.transpose();

            // solve lasso for new mu
            optimization::StateLassoFull<colmat_value_t> state_lasso(
                hess, _penalty, _nnls_max_iters, _nnls_tol, mu, grad
            );
            optimization::lasso_full(state_lasso); 
        }

        throw util::adelie_core_solver_error("ConstraintBoxProximalNewton: max iterations reached!");
    }
};

} // namespace constraint
} // namespace adelie_core