#pragma once
#include <adelie_core/constraint/constraint_base.hpp>
#include <adelie_core/bcd/unconstrained/newton.hpp>
#include <adelie_core/optimization/nnqp_full.hpp>
#include <adelie_core/util/macros.hpp>

namespace adelie_core {
namespace constraint {

template <class ValueType>
class ConstraintLowerUpper: public ConstraintBase<ValueType>
{
public:
    using base_t = ConstraintBase<ValueType>;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using typename base_t::colmat_value_t;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;

private:
    const value_t _sgn;
    const map_cvec_value_t _b;
    const size_t _max_iters;
    const value_t _tol;
    const size_t _newton_max_iters = 100000;
    const value_t _newton_tol = 1e-12;
    const size_t _nnls_max_iters;
    const value_t _nnls_tol;
    const value_t _slack;
    vec_value_t _buff;

#ifdef ADELIE_CORE_DEBUG
    std::vector<vec_value_t> _primals;
    std::vector<vec_value_t> _duals;
#endif

public:
    explicit ConstraintLowerUpper(
        value_t sgn,
        const Eigen::Ref<const vec_value_t> b,
        size_t max_iters,
        value_t tol,
        size_t nnls_max_iters,
        value_t nnls_tol,
        value_t slack
    ):
        _sgn(sgn),
        _b(b.data(), b.size()),
        _max_iters(max_iters),
        _tol(tol),
        _nnls_max_iters(nnls_max_iters),
        _nnls_tol(nnls_tol),
        _slack(slack),
        _buff((b.size() <= 1) ? 0 : (b.size() * (9 + 2 * b.size())))
    {
        if (std::abs(sgn) != 1) {
            throw util::adelie_core_error("sgn must be +/-1.");
        }
        if ((b < 0).any()) {
            throw util::adelie_core_error("b must be >= 0.");
        }
        if (tol < 0) {
            throw util::adelie_core_error("tol must be >= 0.");
        }
        if (nnls_tol < 0) {
            throw util::adelie_core_error("nnls_tol must be >= 0.");
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
            const auto b = _b[0];
            const auto q = quad[0];
            const auto v = linear[0];

            // check if x == 0 is optimal
            auto mu0 = (b > 0) ? 0 : std::max<value_t>(_sgn * v, 0.0);
            const auto is_zero_opt = std::abs(v - _sgn * mu0) <= l1;

            // if optimal, take previous solution, else compute general solution
            const auto x0 = is_zero_opt ? 0 : (
                _sgn * std::min(
                    _sgn * std::copysign(std::abs(v) - l1, v) / (q + l2), 
                    b
                )
            );
            mu0 = is_zero_opt ? mu0 : (
                (_sgn * x0 < b) ? 0 : (_sgn * (v - ((q + l2) * x0 + std::copysign(l1, x0))))
            );

            // store output
            x[0] = x0;
            mu[0] = mu0;

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

        const auto compute_mu_resid = [&]() {
            mu_resid.matrix() = v.matrix() - _sgn * mu.matrix() * Q;
        };
        const auto compute_primal = [&]() {
            size_t x_iters;
            bcd::unconstrained::newton_abs_solver(
                quad, mu_resid, l1, l2, _newton_tol, _newton_max_iters, 
                x, x_iters, x_buffer1, x_buffer2
            );
        };

        bool is_prev_valid = false;
        bool zero_primal_checked = false;
        value_t mu_resid_norm_sq_prev = -1;

        while (iters < _max_iters) {
            ++iters;

            compute_mu_resid();
            const value_t mu_resid_norm_sq = mu_resid.square().sum();

            // if x^star(mu) == 0
            if (mu_resid_norm_sq <= l1 * l1) {
                // Check if there is a primal-dual optimal pair where primal = 0.
                // To be optimal, they must satisfy the 4 KKT conditions.
                if (!zero_primal_checked) {
                    zero_primal_checked = true;

                    Qv.matrix() = v.matrix() * Q.transpose();
                    auto& mu_curr = alpha;
                    if (is_prev_valid) mu_curr = mu; // optimization
                    mu = (_sgn * Qv).max(0) * (_b <= 0).template cast<value_t>();

                    // if (0, mu) is optimal
                    const value_t nnls_loss = (Qv - _sgn * mu).square().sum();
                    if (nnls_loss <= l1 * l1) {
                        x.setZero();
                        #ifdef ADELIE_CORE_DEBUG
                        _primals.push_back(x);
                        _duals.push_back(mu);
                        if (Eigen::isnan(mu).any()) {
                            throw util::adelie_core_error("Found nan! 196");
                        }
                        #endif
                        return;
                    }

                    if (is_prev_valid) mu = mu_curr; // optimization
                }

                // If we ever enter this region of code, it means that
                // there is no primal-dual optimal pair where primal = 0.
                // This must mean that if there is no previous mu,
                // we must have come from the if-block above, in which case nnls_loss > l1 * l1
                // so the next iteration with the current mu will give a non-zero primal (start proximal Newton from here);
                // otherwise, the proximal newton step overshot so we must backtrack.
                if (is_prev_valid) {
                    const value_t lmda_target = (1-_slack) * l1 + _slack * mu_resid_norm_sq_prev;
                    const value_t a = (mu - mu_prev).square().sum();
                    const value_t b = _sgn * ((Qv - _sgn * mu) * (mu - mu_prev)).sum();
                    const value_t c = mu_resid_norm_sq - lmda_target * lmda_target;
                    const value_t t_star = (-b + std::sqrt(std::max<value_t>(b * b - a * c, 0.0))) / a;
                    const value_t step_size = std::max<value_t>(1-t_star, 0.0);
                    mu = mu_prev + step_size * (mu - mu_prev);
                }
                continue;
            }

            compute_primal();

            #ifdef ADELIE_CORE_DEBUG
            _primals.push_back(x);
            _duals.push_back(mu);
            if (Eigen::isnan(mu).any()) {
                throw util::adelie_core_error("Found nan! 225");
            }
            #endif

            grad.matrix() = _sgn * x.matrix() * Q.transpose() - _b.matrix();

            if (is_prev_valid) {
                const auto convg_meas = std::abs(
                    ((mu-mu_prev) * (grad_prev-grad)).sum()
                ) / m;
                if (convg_meas <= _tol) return;
            }

            // Compute hessian
            // NOTE:
            //  - x_buffer1 = quad + l2
            //  - x_buffer2 = 1 / (x_buffer1 * x_norm + lmda)

            hess.setZero();
            auto hess_lower = hess.template selfadjointView<Eigen::Lower>();

            // lower(hess) += x_norm * Q diag(x_buffer2) Q^T 
            hess_buff.array() = Q.array().rowwise() * x_buffer2.sqrt();
            const auto x_norm = x.matrix().norm();
            hess_lower.rankUpdate(hess_buff, x_norm);

            // lower(hess) += x_norm * lmda * kappa * alpha alpha^T
            alpha_tmp = (x * x_buffer2) / x_norm;
            alpha.matrix() = alpha_tmp.matrix() * Q.transpose();
            const auto l1_kappa_norm = l1 * x_norm / (x * x_buffer1 * alpha_tmp).sum();
            hess_lower.rankUpdate(alpha.matrix().transpose(), l1_kappa_norm);

            // full hessian update
            hess.template triangularView<Eigen::Upper>() = hess.transpose();

            // save old values
            mu_resid_norm_sq_prev = mu_resid_norm_sq;
            mu_prev = mu;
            grad_prev = grad;
            is_prev_valid = true;

            // solve NNQP for new mu
            optimization::StateNNQPFull<colmat_value_t> state_nnqp(
                hess, _nnls_max_iters, _nnls_tol, 0, mu, grad
            );
            optimization::nnqp_full(state_nnqp); 
        }

        compute_mu_resid();
        compute_primal();
        #ifdef ADELIE_CORE_DEBUG
        _primals.push_back(x);
        _duals.push_back(mu);
        if (Eigen::isnan(mu).any()) {
            throw util::adelie_core_error("Found nan! 279");
        }
        #endif
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

} // namespace constraint
} // namespace adelie_core