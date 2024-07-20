#pragma once
#include <adelie_core/bcd/unconstrained/newton.hpp>
#include <adelie_core/optimization/newton.hpp>
#include <adelie_core/util/types.hpp>

namespace adelie_core {
namespace bcd {
namespace constrained {

template <class QuadType, class LinearType, class ValueType,
          class AType, class BType, class AVarType,
          class OutType, class BuffType>
void coordinate_descent_solver(
    const QuadType& quad,
    const LinearType& linear,
    ValueType l1,
    ValueType l2,
    const AType& A,
    const BType& b,
    const AVarType& A_vars,
    size_t max_iters,
    ValueType tol,
    size_t pnewton_max_iters,
    ValueType pnewton_tol,
    size_t newton_max_iters,
    ValueType newton_tol,
    size_t& iters,
    OutType& x,
    OutType& mu,
    OutType& mu_resid,
    ValueType& mu_rsq,
    BuffType& buff
)
{
    using value_t = ValueType;
    using vec_value_t = util::rowvec_type<value_t>;

    if (l1 <= 0) {
        // TODO: this is just QP with linear inequality constraint.
        throw util::adelie_core_error(
            "coordinate_descent_solver: case l1 <= 0 not implemented yet!"
        );
    }

    const auto m = A.rows();
    const auto d = A.cols();
    const auto& S = quad;
    const auto& v = linear;

    iters = 0;

    Eigen::Map<vec_value_t> x_buffer1(buff.data(), d);
    Eigen::Map<vec_value_t> x_buffer2(buff.data()+d, d);

    // optimization: check if unconstrained solution is feasible.
    {
        Eigen::Map<vec_value_t> x_uncnstr(buff.data()+2*d, d);
        size_t x_iters;
        unconstrained::newton_solver(
            quad, linear, l1, l2, newton_tol, newton_max_iters, 
            x_uncnstr, x_iters, x_buffer1, x_buffer2
        );
        x_buffer1.matrix() = x_uncnstr.matrix() * A.transpose();
        // if unconstrained solution is feasible
        if ((x_buffer1 <= b).all()) {
            x = x_uncnstr;
            mu.setZero();
            mu_resid = v;
            mu_rsq = v.square().sum();
            return;
        }
    }

    // since unconstrained solution is wrong, 

    /* invariance quantities */
    // mu_resid = v - A.T @ mu
    // mu_rsq = ||mu_resid||_2^2

    value_t convg_measure = 0;

    const auto invariance_f = [&](
        size_t k, value_t mu_k, value_t Akr
    ) {
        const auto del = mu_k - mu[k];
        mu[k] = mu_k;
        mu_resid -= del * A.row(k).array();
        mu_rsq -= del * (2 * Akr - A_vars[k] * del);
    };

    while (iters < max_iters) {
        bool compute_x = false;
        convg_measure = 0;
        ++iters;

        // coordinate descent
        for (int k = 0; k < m; ++k) {
            const auto A_vars_k = A_vars[k];

            // if kth constraint can be removed.
            if (A_vars_k <= 0) {
                mu[k] = 0;
                continue;
            }

            // compute l_star
            auto Akr = A.row(k).dot(mu_resid.matrix());
            const auto discr = Akr * Akr - A_vars_k * (mu_rsq - l1 * l1);
            const bool l_star_finite = discr >= 0;
            value_t l_star = l_star_finite ? 
                (mu[k] + (Akr - std::sqrt(discr)) / A_vars_k) :
                std::numeric_limits<value_t>::max()
            ;

            /* case 1: b_k = 0 and l_star < infty */
            if (b[k] == 0 && l_star_finite) {
                const auto mu_k_new = std::max<value_t>(l_star, 0);
                if (mu_k_new == mu[k]) continue; 
                invariance_f(k, mu_k_new, Akr);
                const auto del = mu_k_new - mu[k];
                convg_measure = std::max(convg_measure, A_vars[k] * del * del);
                compute_x = true;
                continue;
            }

            // optimization: l_star <= 0 (and b_k > 0)
            if (l_star <= 0) {
                if (mu[k] == 0) continue;
                invariance_f(k, 0, Akr);
                const auto del = -mu[k];
                convg_measure = std::max(convg_measure, A_vars[k] * del * del);
                compute_x = true;
                continue;
            }

            /* case 2: b_k > 0 and 0 < l_star < infty */
            /* case 3: l_star = infty */

            // compute x
            size_t x_iters;
            unconstrained::newton_solver(
                quad, mu_resid, l1, l2, newton_tol, newton_max_iters, 
                x, x_iters, x_buffer1, x_buffer2
            );
            compute_x = false;

            // optimization: k is inactive and will stay inactive
            if (mu[k] == 0) {
                const auto h_k = A.row(k).dot(x.matrix()) - b[k];
                if (h_k <= 0) continue;
            }

            /* projected Newton method */

            // NOTE: the following are additional invariance quantities from now:
            // - x
            // - Akr
            const auto mu_k_old = mu[k];
            const auto initial_f = [&]() {
                return std::make_tuple(mu[k], 0);
            };
            bool is_first_call = true;
            const auto step_f = [&](auto muk) {
                if (is_first_call) {
                    is_first_call = false;
                } else {
                    invariance_f(k, muk, Akr);
                    unconstrained::newton_solver(
                        quad, mu_resid, l1, l2, newton_tol, newton_max_iters, 
                        x, x_iters, x_buffer1, x_buffer2
                    );
                    Akr = A.row(k).dot(mu_resid.matrix());
                }
                const auto x_norm = x.matrix().norm();

                // NOTE: this really should not happen, but just in case..
                if (x_norm <= 0) {
                    const value_t fh = -b[k];
                    const value_t dfh = -Akr * Akr / ((S + l2) * mu_resid.square()).sum();
                    return std::make_tuple(fh, dfh);
                }

                // Since x_norm > 0 and l1 > 0,
                //x_buffer1 = S + l2
                //x_buffer2 = 1 / (x_buffer1 * x_norm + l1)

                // compute intermediate values
                Eigen::Map<vec_value_t> t1(buff.data()+2*d, d);
                Eigen::Map<vec_value_t> t2(buff.data()+3*d, d);
                t1 = A.row(k).array() * x_buffer2;
                t2 = mu_resid * x_buffer2;
                const auto t3 = (t1 * t2).sum();

                // compute output
                const value_t fh = x_norm * (t1 * mu_resid).sum() - b[k];
                const value_t dfh = -(
                    x_norm * (t1 * A.row(k).array()).sum()
                    + l1 * t3 * t3 / (x_buffer1 * t2.square() * x_buffer2).sum()
                );
                return std::make_tuple(fh, dfh);
            };
            const auto project_f = [&](auto h) { 
                return std::max<value_t>(std::min<value_t>(h, l_star), 0); 
            };
            optimization::newton_root_find(
                initial_f,
                step_f,
                project_f,
                pnewton_tol,
                pnewton_max_iters
            );

            const auto del = mu[k] - mu_k_old;
            convg_measure = std::max(convg_measure, A_vars[k] * del * del);
        }

        // check convergence
        if (convg_measure <= tol) {
            if (compute_x) {
                size_t x_iters;
                unconstrained::newton_solver(
                    quad, mu_resid, l1, l2, newton_tol, newton_max_iters, 
                    x, x_iters, x_buffer1, x_buffer2
                );
            }
            break;
        }
    }    
}

} // namespace constrained
} // namespace bcd
} // namespace adelie_core