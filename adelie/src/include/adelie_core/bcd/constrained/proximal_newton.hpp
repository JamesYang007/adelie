#pragma once
#include <adelie_core/bcd/unconstrained/newton.hpp>
#include <adelie_core/optimization/nnls.hpp>
#include <adelie_core/util/types.hpp>

namespace adelie_core {
namespace bcd {
namespace constrained {

template <class QuadType, class LinearType, class ValueType,
          class AType, class BType, class ATVarsType,
          class OutType, class BuffType>
void proximal_newton_solver(
    const QuadType& quad,
    const LinearType& linear,
    ValueType l1,
    ValueType l2,
    const AType& A,
    const BType& b,
    const ATVarsType& AT_vars,
    size_t max_iters,
    ValueType tol,
    size_t newton_max_iters,
    ValueType newton_tol,
    size_t nnls_max_iters,
    ValueType nnls_tol,
    ValueType nnls_dtol,
    size_t& iters,
    OutType& x,
    OutType& mu,
    OutType& mu_resid,
    BuffType& buff
)
{
    using value_t = ValueType;
    using vec_value_t = util::rowvec_type<value_t>;
    using colmat_value_t = util::colmat_type<value_t>;
    using rowmat_value_t = util::rowmat_type<value_t>;

    // TODO: optimization: check if (0, mu) is valid pair.
    // TODO: must check for l1 <= 0 case! 
    // Main loop logic assumes x_buffer1, x_buffer2 are filled.

    const auto m = A.rows();
    const auto d = A.cols();
    const auto& S = quad;
    const auto& v = linear;

    iters = 0;
    
    auto buff_ptr = buff.data();
    Eigen::Map<vec_value_t> mu_prev(buff_ptr, m); buff_ptr += m;
    Eigen::Map<vec_value_t> grad(buff_ptr, m); buff_ptr += m;
    Eigen::Map<vec_value_t> grad_prev(buff_ptr, m); buff_ptr += m;
    Eigen::Map<colmat_value_t> hess(buff_ptr, m, m); buff_ptr += m * m;
    Eigen::Map<rowmat_value_t> hess_buff(buff_ptr, m, d); buff_ptr += m * d;
    Eigen::Map<vec_value_t> x_buffer1(buff_ptr, d); buff_ptr += d;
    Eigen::Map<vec_value_t> x_buffer2(buff_ptr, d); buff_ptr += d;
    Eigen::Map<vec_value_t> alpha(buff_ptr, m); buff_ptr += m;
    Eigen::Map<vec_value_t> alpha_tmp(buff_ptr, d); buff_ptr += d;
    bool is_prev_valid = false;
    bool recompute_mu_resid = true;

    while (iters < max_iters) {
        ++iters;

        // compute v - A^T mu
        if (recompute_mu_resid) {
            mu_resid.matrix().noalias() = mu.matrix() * A;
            mu_resid = v - mu_resid;
        }

        // compute x^star(mu)
        {
            size_t x_iters;
            unconstrained::newton_abs_solver(
                quad, mu_resid, l1, l2, newton_tol, newton_max_iters, 
                x, x_iters, x_buffer1, x_buffer2
            );
        }
        const auto x_norm = x.matrix().norm();

        // If ||x||_2 = 0, check if 0 is a valid solution.
        if (x_norm <= 0) {
            for (int i = 0; i < m; ++i) {
                if (b[i] <= 0 || mu[i] <= 0) continue;
                mu_resid += mu[i] * A.row(i).array();
                mu[i] = 0;
            }
            size_t nnls_iters;
            value_t mu_loss = 0.5 * mu_resid.square().sum();
            optimization::nnls_naive(
                A.transpose(), AT_vars, nnls_iters, nnls_tol, nnls_dtol,
                nnls_iters, mu, mu_resid, mu_loss,
                [&]() { return mu_loss <= 0.5 * l1; },
                [&](auto i) { return b[i] > 0; }
            );

            // If mu_loss is smaller than or close to l1, 
            // then check passed and 0 is a valid primal solution.
            if (mu_loss <= l1 * (0.5+5e-5)) return;

            // Otherwise, 0 is probably not the solution.
            // The proximal operator simplifies to solving:
            //      minimize_{mu >= 0} -b^T mu
            // This means we set mu[i] = 0 whenever b[i] > 0, 
            // and mu_resid must be updated accordingly.
            // Note that this was exactly done beforehand!
            // Now, just continue with the updated mu :).
            recompute_mu_resid = false;
            continue;
        }

        // compute (negative) gradient
        grad.matrix().noalias() = x.matrix() * A.transpose();
        grad -= b;

        // check convergence
        if (is_prev_valid) {
            const auto convg_meas = std::abs(
                ((mu-mu_prev) * (grad_prev-grad)).sum()
            ) / m;
            if (convg_meas <= tol) break;
        }

        // compute hessian
        // x_buffer1 = quad + l2
        // x_buffer2 = x_buffer1 * x_norm + lmda
        hess_buff.array() = A.array().rowwise() / x_buffer2;
        hess.noalias() = hess_buff * A.transpose();
        alpha_tmp = x / (x_norm * x_buffer2);
        alpha.matrix().noalias() = alpha_tmp.matrix() * A.transpose();
        const auto kappa = 1 / (1 - l1 * (x * alpha_tmp).sum() / x_norm);
        hess.noalias() += l1 * kappa * alpha.matrix().transpose() * alpha.matrix();
        hess.array() *= x_norm;

        // save old values
        mu_prev = mu;
        grad_prev = grad;
        is_prev_valid = true;

        // solve NNLS for new mu
        size_t nnls_iters = 0;
        value_t loss = 0;
        optimization::nnls_cov_full(
            hess, nnls_max_iters, nnls_tol, nnls_dtol,
            nnls_iters, mu, grad, loss, [](){return false;}
        ); 
        recompute_mu_resid = true;
    }
}

} // namespace constrained
} // namespace bcd
} // namespace adelie_core