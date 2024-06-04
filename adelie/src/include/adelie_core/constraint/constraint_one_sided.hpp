#pragma once
#include <adelie_core/constraint/constraint_base.hpp>
#include <adelie_core/bcd/unconstrained/newton.hpp>
#include <adelie_core/optimization/nnqp_full.hpp>
#include <adelie_core/util/macros.hpp>

namespace adelie_core {
namespace constraint {

template <class ValueType>
class ConstraintOneSidedBase: public ConstraintBase<ValueType>
{
public:
    using base_t = ConstraintBase<ValueType>;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using typename base_t::colmat_value_t;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;

protected:
    const map_cvec_value_t _sgn;
    const map_cvec_value_t _b;

public:
    explicit ConstraintOneSidedBase(
        const Eigen::Ref<const vec_value_t> sgn,
        const Eigen::Ref<const vec_value_t> b
    ):
        _sgn(sgn.data(), sgn.size()),
        _b(b.data(), b.size())
    {
        if ((_sgn.abs() != 1).any()) {
            throw util::adelie_core_error("sgn must be a vector of +/-1.");
        }
        if ((_b < 0).any()) {
            throw util::adelie_core_error("b must be >= 0.");
        }
        if (_sgn.size() != _b.size()) {
            throw util::adelie_core_error("sgn and b must have the same length.");
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

    void project(
        Eigen::Ref<vec_value_t> x
    ) override
    {
        x = _sgn * (_sgn * x).min(_b);
    }

    void gradient(
        const Eigen::Ref<const vec_value_t>&,
        const Eigen::Ref<const vec_value_t>& mu,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        out = _sgn * mu;
    }

    int duals() override { return _b.size(); }
    int primals() override { return _b.size(); }
};

template <class ValueType>
class ConstraintOneSidedProximalNewton: public ConstraintOneSidedBase<ValueType>
{
public:
    using base_t = ConstraintOneSidedBase<ValueType>;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using typename base_t::colmat_value_t;
    using typename base_t::map_cvec_value_t;
    using base_t::_sgn;
    using base_t::_b;

private:
    const size_t _max_iters;
    const value_t _tol;
    const size_t _newton_max_iters = 100000;
    const value_t _newton_tol = 1e-12;
    const size_t _nnls_max_iters;
    const value_t _nnls_tol;
    const value_t _cs_tol;
    const value_t _slack;
    vec_value_t _buff;

#ifdef ADELIE_CORE_DEBUG
    std::vector<vec_value_t> _primals;
    std::vector<vec_value_t> _duals;
#endif

public:
    explicit ConstraintOneSidedProximalNewton(
        const Eigen::Ref<const vec_value_t> sgn,
        const Eigen::Ref<const vec_value_t> b,
        size_t max_iters,
        value_t tol,
        size_t nnls_max_iters,
        value_t nnls_tol,
        value_t cs_tol,
        value_t slack
    ):
        base_t(sgn, b),
        _max_iters(max_iters),
        _tol(tol),
        _nnls_max_iters(nnls_max_iters),
        _nnls_tol(nnls_tol),
        _cs_tol(cs_tol),
        _slack(slack),
        _buff((b.size() <= 1) ? 0 : (b.size() * (9 + 2 * b.size())))
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

    auto debug_info() const 
    {
        #ifdef ADELIE_CORE_DEBUG
        util::rowmat_type<value_t> pr(_primals.size(), _b.size());
        util::rowmat_type<value_t> du(_duals.size(), _b.size());
        for (size_t i = 0; i < _primals.size(); ++i) 
        {
            pr.row(i) = _primals[i];
            du.row(i) = _duals[i];
        }
        return std::make_tuple(pr, du); 
        #endif
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

        #ifdef ADELIE_CORE_DEBUG
        _primals.clear();
        _duals.clear();
        #endif
        
        const auto m = _b.size();
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
            mu_resid.matrix() = v.matrix() - (_sgn * mu).matrix() * Q;
        };
        const auto compute_primal = [&]() {
            size_t x_iters;
            bcd::unconstrained::newton_abs_solver(
                quad, mu_resid, l1, l2, _newton_tol, _newton_max_iters, 
                x, x_iters, x_buffer1, x_buffer2
            );
        };
        #ifdef ADELIE_CORE_DEBUG
        const auto save_iterate = [&]() {
            _primals.push_back(x);
            _duals.push_back(mu);
            if (Eigen::isnan(x).any() || Eigen::isnan(mu).any()) {
                PRINT(x);
                PRINT(mu);
                throw util::adelie_core_error("Found nan!");
            }
        };
        #endif

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
                        ((mu-mu_prev) * (grad_prev+_b)).sum()
                    ) / m;
                    if (convg_meas <= _tol) {
                        x.setZero();
                        #ifdef ADELIE_CORE_DEBUG
                        save_iterate();
                        #endif
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
                        grad_prev = -_b;
                        is_prev_valid = true;
                    }
                    mu = (_sgn * Qv).max(0);

                    // Technically, we must find mu such that:
                    // 1) KKT first-order condition: ||v - Q.T @ (_sgn * mu)||_2 <= l1.
                    // 2) Primal feasibility: _sgn * Q @ x <= b (already satisfied with x = 0).
                    // 3) Dual feasibility: mu >= 0.
                    // 4) Complementary slackness: mu * _b = 0.
                    // Perform 2 checks:
                    // a) Relax 4) by setting mu = (_sgn * Q @ v) to minimize the norm in 1)
                    //    and checking whether mean((mu * _b) ** 2) is small.
                    // b) Mathematically, mu = (_sgn * Q @ v) * (_b <= 0) satisfies 2)-4)
                    //    and minimizes the norm in 1). If the norm is <= l1, done.   
                    value_t nnls_loss = (Qv - _sgn * mu).square().sum();
                    if (
                        (nnls_loss <= l1 * l1) &&
                        ((mu * _b).square().mean() <= _cs_tol)
                     ) {
                        x.setZero();
                        #ifdef ADELIE_CORE_DEBUG
                        save_iterate();
                        #endif
                        return;
                    }

                    mu *= (_b <= 0).template cast<value_t>();
                    nnls_loss = (Qv - _sgn * mu).square().sum();
                    if (nnls_loss <= l1 * l1) {
                        x.setZero();
                        #ifdef ADELIE_CORE_DEBUG
                        save_iterate();
                        #endif
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
                const value_t b = ((_sgn * Qv - mu) * (mu - mu_prev)).sum();
                const value_t c = mu_resid_norm_sq - lmda_target * lmda_target;
                const value_t t_star = (-b + std::sqrt(std::max<value_t>(b * b - a * c, 0.0))) / a;
                const value_t step_size = std::min<value_t>(std::max<value_t>(1-t_star, 0.0), 1.0);
                mu = mu_prev + step_size * (mu - mu_prev);
                continue;
            }

            #ifdef ADELIE_CORE_DEBUG
            save_iterate();
            #endif

            grad.matrix() = (x.matrix() * Q.transpose()).cwiseProduct(_sgn.matrix()) - _b.matrix();

            // optimization: if optimality hard-check is satisfied, finish early.
            if ((grad <= 0).all() && (mu * grad == 0).all()) return;

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

            // reparametrize
            grad *= _sgn;
            mu *= _sgn;

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

            // solve NNQP for new mu
            optimization::StateNNQPFull<colmat_value_t, true> state_nnqp(
                _sgn, hess, _nnls_max_iters, _nnls_tol, mu, grad
            );
            optimization::nnqp_full(state_nnqp); 

            // reparametrize
            mu *= _sgn;
        }

        throw util::adelie_core_solver_error("ConstraintOneSidedProximalNewton: max iterations reached!");
    }
};

template <class ValueType>
class ConstraintOneSidedADMM: public ConstraintOneSidedBase<ValueType>
{
public:
    using base_t = ConstraintOneSidedBase<ValueType>;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using typename base_t::colmat_value_t;
    using typename base_t::map_cvec_value_t;
    using base_t::_sgn;
    using base_t::_b;

private:
    const size_t _max_iters;
    const value_t _tol_abs;
    const value_t _tol_rel;
    const size_t _newton_max_iters = 100000;
    const value_t _newton_tol = 1e-12;
    const value_t _rho;
    vec_value_t _buff;

#ifdef ADELIE_CORE_DEBUG
    std::vector<vec_value_t> _primals1;
    std::vector<vec_value_t> _primals2;
    std::vector<vec_value_t> _duals;
#endif

public:
    explicit ConstraintOneSidedADMM(
        const Eigen::Ref<const vec_value_t> sgn,
        const Eigen::Ref<const vec_value_t> b,
        size_t max_iters,
        value_t tol_abs,
        value_t tol_rel,
        value_t rho
    ):
        base_t(sgn, b),
        _max_iters(max_iters),
        _tol_abs(tol_abs),
        _tol_rel(tol_rel),
        _rho(rho),
        _buff((b.size() <= 1) ? 0 : (7 * b.size()))
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

    auto debug_info() const 
    {
        #ifdef ADELIE_CORE_DEBUG
        util::rowmat_type<value_t> pr1(_primals1.size(), _b.size());
        util::rowmat_type<value_t> pr2(_primals2.size(), _b.size());
        util::rowmat_type<value_t> du(_duals.size(), _b.size());
        for (size_t i = 0; i < _primals1.size(); ++i) 
        {
            pr1.row(i) = _primals1[i];
            pr2.row(i) = _primals2[i];
            du.row(i) = _duals[i];
        }
        return std::make_tuple(pr1, pr2, du); 
        #endif
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
        #ifdef ADELIE_CORE_DEBUG
        _primals1.clear();
        _primals2.clear();
        _duals.clear();
        #endif
        
        const auto m = _b.size();
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
        Eigen::Map<vec_value_t> z(buff_ptr, m); buff_ptr += m;
        Eigen::Map<vec_value_t> z_prev(buff_ptr, m); buff_ptr += m;
        Eigen::Map<vec_value_t> u(buff_ptr, m); buff_ptr += m;
        Eigen::Map<vec_value_t> linear_shifted(buff_ptr, d); buff_ptr += d;
        Eigen::Map<vec_value_t> s(buff_ptr, m); buff_ptr += m;

        // TODO: these warm-starts may not be optimal
        z = ((x.matrix() * Q.transpose()).array() * _sgn).min(_b);
        u.setZero();

        const auto compute_primal_x = [&]() {
            size_t x_iters;
            bcd::unconstrained::newton_abs_solver(
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
                mu = (
                    ((v.matrix() * Q.transpose()).array() * _sgn).max(0) 
                    * (_b <= 0).template cast<value_t>()
                );
                return;
            }
            mu = (((v - (quad + l2 + l1 / x_norm) * x).matrix() * Q.transpose()).array() * _sgn).max(0);
        };

        #ifdef ADELIE_CORE_DEBUG
        const auto save_iterate = [&]() {
            _primals1.push_back(x);
            _primals2.push_back(z);
            _duals.push_back(mu);
            if (Eigen::isnan(x).any() || Eigen::isnan(z).any() || Eigen::isnan(mu).any()) {
                PRINT(x);
                PRINT(z);
                PRINT(mu);
                throw util::adelie_core_error("Found nan!");
            }
        };
        #endif

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
};

} // namespace constraint
} // namespace adelie_core