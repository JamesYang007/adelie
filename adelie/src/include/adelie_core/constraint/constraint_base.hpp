#pragma once
#include <cstdio>
#include <adelie_core/bcd/unconstrained/newton.hpp>
#include <adelie_core/util/exceptions.hpp>
#include <adelie_core/util/format.hpp>
#include <adelie_core/util/macros.hpp>
#include <adelie_core/util/types.hpp>
#include <adelie_core/util/stopwatch.hpp>

namespace adelie_core {
namespace constraint {

template <class ValueType, class IndexType=Eigen::Index>
class ConstraintBase
{
public:
    using value_t = ValueType;
    using index_t = IndexType;
    using vec_index_t = util::rowvec_type<index_t>;
    using vec_value_t = util::rowvec_type<value_t>;
    using vec_uint64_t = util::rowvec_type<uint64_t>;
    using colmat_value_t = util::colmat_type<value_t>;

protected:
    static void check_solve(
        int x, int q, int l, int m, int d
    ) {
        if (
            (x != q) ||
            (q != l) ||
            (l != d)
        ) {
            throw util::adelie_core_error(
                util::format(
                    "solve() is given inconsistent inputs! "
                    "Invoked solve(x=%d, q=%d, l=%d, m=%d, d=%d)",
                    x, q, l, m, d
                )
            );
        }
    }

    static void check_gradient(
        int x, int mu, int o, int m, int d
    ) {
        if (
            (x != o) ||
            (o != d) ||
            (mu != m)
        ) {
            throw util::adelie_core_error(
                util::format(
                    "gradient() is given inconsistent inputs! "
                    "Invoked gradient(x=%d, mu=%d, o=%d, m=%d, d=%d)",
                    x, mu, o, m, d
                )
            );
        }
    }

    // NOTE: this only works for linear inequality constraint!
    template <class ComputeMuResidType,
              class ComputeMinMuResidType,
              class ComputeBacktrackAType,
              class ComputeBacktrackBType,
              class ComputeBacktrackType,
              class ComputeGradientType,
              class ComputeHardOptimalityType,
              class ComputeConvergenceMeasureType,
              class ComputeProximalNewtonStepType,
              class SaveAdditionalPrevType>
    void _solve_proximal_newton(
        Eigen::Ref<vec_value_t> x,
        const Eigen::Ref<const vec_value_t>& quad,
        const Eigen::Ref<const vec_value_t>& linear,
        value_t l1,
        value_t l2,
        const Eigen::Ref<const colmat_value_t>& Q,
        size_t max_iters,
        value_t tol,
        value_t slack,
        vec_uint64_t buff,
        ComputeMuResidType compute_mu_resid,
        ComputeMinMuResidType compute_min_mu_resid,
        ComputeBacktrackAType compute_backtrack_a,
        ComputeBacktrackBType compute_backtrack_b,
        ComputeBacktrackType compute_backtrack,
        ComputeGradientType compute_gradient,
        ComputeHardOptimalityType compute_hard_optimality,
        ComputeConvergenceMeasureType compute_convergence_measure,
        ComputeProximalNewtonStepType compute_proximal_newton_step,
        SaveAdditionalPrevType save_additional_prev
    )
    {
        using rowmat_value_t = util::rowmat_type<value_t>;

        const auto d = x.size();

        const auto& v = linear;

        size_t iters = 0;

        // size must be at least d * (6 + 2 * d)
        auto buff_ptr = reinterpret_cast<value_t*>(buff.data());
        Eigen::Map<vec_value_t> x_buffer1(buff_ptr, d); buff_ptr += d;
        Eigen::Map<vec_value_t> x_buffer2(buff_ptr, d); buff_ptr += d;
        Eigen::Map<vec_value_t> mu_resid(buff_ptr, d); buff_ptr += d;
        Eigen::Map<rowmat_value_t> hess_buff(buff_ptr, d, d); buff_ptr += d * d;
        Eigen::Map<colmat_value_t> hess(buff_ptr, d, d); buff_ptr += d * d;
        Eigen::Map<vec_value_t> alpha_tmp(buff_ptr, d); buff_ptr += d;
        Eigen::Map<vec_value_t> alpha(buff_ptr, d); buff_ptr += d;
        Eigen::Map<vec_value_t> Qv(buff_ptr, d); buff_ptr += d;

        Qv.matrix() = v.matrix() * Q.transpose();

        bool is_x_init_zero = (x == 0).all();
        bool is_prev_valid = false;
        bool zero_primal_checked = false;
        value_t mu_resid_norm_prev = -1;

        // If x is initialized to be 0, we assume it is a good warm-start
        // so that it is likely that the next update is also 0.
        // In this case, we first check if the dual can be updated such that x remains to be 0.
        if (is_x_init_zero) {
            zero_primal_checked = true;
            if (compute_min_mu_resid(Qv, false, true) <= l1 * l1) {
                return;
            }
        }

        const auto compute_primal = [&]() {
            constexpr size_t _newton_max_iters = 100000;
            constexpr value_t _newton_tol = 1e-12;
            size_t x_iters;
            bcd::unconstrained::newton_solver(
                quad, mu_resid, l1, l2, _newton_tol, _newton_max_iters, 
                x, x_iters, x_buffer1, x_buffer2
            );
        };

        while (iters < max_iters) {
            ++iters;

            compute_mu_resid(mu_resid);
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
                // If x was initially zero and in the first iteration we ended up with x == 0 again
                // (likely from numerical precision reasons), just return 0.
                if (iters == 1 && is_x_init_zero) {
                    x.setZero(); 
                    return;
                }

                // NOTE: this check is important since numerical precision issues
                // may make us enter this loop infinitely.
                if (is_prev_valid) {
                    if (compute_convergence_measure(true) <= tol) {
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

                    const bool is_prev_valid_old = is_prev_valid;
                    if (!is_prev_valid_old) {
                        mu_resid_norm_prev = mu_resid_norm;
                        is_prev_valid = true;
                        save_additional_prev(true);
                    }

                    // Technically, we must find mu such that:
                    // 1) KKT first-order condition: ||v - Q.T @ (A.T @ mu)||_2 <= l1.
                    // 2) Primal feasibility: A @ Q @ x <= b (already satisfied with x = 0).
                    // 3) Dual feasibility: mu >= 0.
                    // 4) Complementary slackness: mu * b = 0.
                    //
                    // Relax 4) to mu * b <= cs_tol and minimize 1) residual norm.
                    // This effectively puts a box-constraint on mu.
                    if (
                        compute_min_mu_resid(Qv, is_prev_valid_old, false) <= l1 * l1
                     ) {
                        x.setZero();
                        return;
                    }

                    if (!is_prev_valid_old) continue;
                }

                // If we ever enter this region of code, it means that
                // there is no primal-dual optimal pair where primal = 0.
                // The proximal newton step overshot so we must backtrack.
                if (!is_prev_valid || (mu_resid_norm_prev <= l1 * 0.9999) || (mu_resid_norm > l1 * 1.0001)) {
                    throw util::adelie_core_error(
                        "Possibly an unexpected error! "
                        "Previous iterate should have been properly initialized. "
                        "This may occur if cs_tol is too small. "
                        "If increasing cs_tol does not fix the issue, "
                        "please report this as a bug! "
                    );
                }
                const value_t lmda_target = (1-slack) * l1 + slack * mu_resid_norm_prev;
                const value_t a = compute_backtrack_a(); 
                const value_t b = compute_backtrack_b(Qv, mu_resid);
                const value_t c = mu_resid_norm_sq - lmda_target * lmda_target;
                const value_t t_star = (-b + std::sqrt(std::max<value_t>(b * b - a * c, 0.0))) / a;
                const value_t step_size = std::min<value_t>(std::max<value_t>(1-t_star, 0.0), 1.0);
                compute_backtrack(step_size);
                continue;
            }

            compute_gradient();

            // optimization: if optimality hard-check is satisfied, finish early.
            if (compute_hard_optimality()) {
                return;
            }

            // Check if mu is not changing much w.r.t. hessian scaling.
            if (is_prev_valid) {
                if (compute_convergence_measure(false) <= tol) {
                    return;
                }
            }

            // save old values
            mu_resid_norm_prev = mu_resid_norm;
            is_prev_valid = true;
            save_additional_prev(false);

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

            compute_proximal_newton_step(hess);
        }

        throw util::adelie_core_solver_error("ConstraintBase: proximal newton max iterations reached!");
    }

public:
    virtual ~ConstraintBase() {}

    virtual void solve(
        Eigen::Ref<vec_value_t> x,
        const Eigen::Ref<const vec_value_t>& quad,
        const Eigen::Ref<const vec_value_t>& linear,
        value_t l1,
        value_t l2,
        const Eigen::Ref<const colmat_value_t>& Q,
        Eigen::Ref<vec_uint64_t> buffer
    ) =0;

    virtual void gradient(
        const Eigen::Ref<const vec_value_t>& x,
        Eigen::Ref<vec_value_t> out
    ) =0;

    // TODO: gradient(x, mu_indices, mu_values, out)
    virtual void gradient(
        const Eigen::Ref<const vec_value_t>& x,
        const Eigen::Ref<const vec_value_t>& mu,
        Eigen::Ref<vec_value_t> out
    ) =0;

    virtual void project(
        Eigen::Ref<vec_value_t>
    )
    {}

    virtual void clear() =0;

    virtual void dual(
        Eigen::Ref<vec_index_t> indices,
        Eigen::Ref<vec_value_t> values
    ) =0;
    virtual int duals_nnz() =0;
    virtual int duals() =0;
    virtual int primals() =0;
    virtual size_t buffer_size() { return 0; }
};

} // namespace constraint
} // namespace adelie_core