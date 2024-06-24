#pragma once
#include <adelie_core/constraint/constraint_base.hpp>
#include <adelie_core/optimization/nnqp_full.hpp>

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
    const size_t _nnls_max_iters;
    const value_t _nnls_tol;
    const value_t _cs_tol;
    const value_t _slack;
    vec_value_t _buff;

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
        const auto m = _b.size();
        const auto d = m;

        if (d == 1) {
            base_t::solve_1d(x, mu, quad, linear, l1, l2, Q);
            return;
        }

        auto buff_ptr = _buff.data();
        Eigen::Map<vec_value_t> grad_prev(buff_ptr, m); buff_ptr += m;
        Eigen::Map<vec_value_t> grad(buff_ptr, m); buff_ptr += m;
        Eigen::Map<vec_value_t> next_buff(buff_ptr, _buff.size() - std::distance(_buff.data(), buff_ptr));

        const auto compute_mu_resid = [&](
            const auto& mu,
            const auto& linear,
            const auto& Q,
            auto& mu_resid
        ) {
            mu_resid.matrix() = linear.matrix() - (_sgn * mu).matrix() * Q;
        };
        const auto compute_hard_min_mu_resid = [&](
            auto& mu,
            const auto& Qv
        ) {
            mu = (_sgn * Qv).max(0);
            return (Qv - _sgn * mu).square().sum();
        };
        const auto compute_soft_min_mu_resid = [&](
            auto& mu,
            const auto& Qv
        ) {
            mu *= (_b <= 0).template cast<value_t>();
            return (Qv - _sgn * mu).square().sum();
        };
        const auto compute_relaxed_slackness = [&](
            const auto& mu
        ) {
            return ((mu * _b).square().mean() <= _cs_tol);
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
            return ((_sgn * Qv - mu) * (mu - mu_prev)).sum();
        };
        const auto compute_gradient = [&](
            const auto& x,
            const auto& Q
        ) {
            grad.matrix() = (x.matrix() * Q.transpose()).cwiseProduct(_sgn.matrix()) - _b.matrix();
        };
        const auto compute_hard_optimality = [&](
            const auto& mu
        ) {
            return (grad <= 0).all() && (mu * grad == 0).all();
        };
        const auto compute_convergence_measure = [&](
            const auto& mu_prev,
            const auto& mu,
            bool is_in_ellipse
        ) {
            return is_in_ellipse ? (
                std::abs(((mu-mu_prev) * (grad_prev+_b)).mean())
            ) : (
                std::abs(((mu-mu_prev) * (grad_prev-grad)).mean())
            );
        };
        const auto compute_proximal_newton_step = [&](
            const auto& hess,
            auto& mu
        ) {
            // reparametrize
            grad *= _sgn;
            mu *= _sgn;

            // solve NNQP for new mu
            optimization::StateNNQPFull<colmat_value_t, true> state_nnqp(
                _sgn, hess, _nnls_max_iters, _nnls_tol, mu, grad
            );
            optimization::nnqp_full(state_nnqp); 

            // reparametrize
            mu *= _sgn;
        };
        const auto save_additional_prev = [&](bool is_in_ellipse) {
            if (is_in_ellipse) grad_prev = -_b;
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