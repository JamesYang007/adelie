#pragma once
#include <atomic>
#include <memory>
#include <adelie_core/solver/solve_basil_base.hpp>
#include <adelie_core/solver/solve_pin_naive.hpp>
#include <adelie_core/util/algorithm.hpp>
#include <adelie_core/util/stopwatch.hpp>

namespace adelie_core {
namespace solver {

/**
 * Checks the KKT condition on the sequence of lambdas and the fitted coefficients.
 * 
 * @param   X       data matrix.
 * @param   y       response vector.
 * @param   alpha   elastic net penalty.
 * @param   lmdas   downward sequence of L1 regularization parameter.
 * @param   betas   each element i corresponds to a (sparse) vector
 *                  of the solution at lmdas[i].
 * @param   is_strong   a functor that checks if feature i is strong.
 * @param   n_threads   number of threads to use in OpenMP.
 * @param   grad        a dense vector that represents the X^T (y - X * beta)
 *                      right __before__ the first value in lmdas (going downwards)
 *                      that fails the KKT check.
 *                      If KKT fails at the first lambda, grad is unchanged.
 * @param   grad_next   a dense vector that represents X^T (y - X * beta)
 *                      right __at__ the first value in lmdas (going downwards)
 *                      that fails the KKT check.
 *                      This is really used for optimizing memory allocation.
 *                      User should not need to access this directly.
 *                      It is undefined-behavior accessing this after the call.
 *                      It just has to be initialized to the same size as grad.
 * @param   abs_grad    similar to grad but represents ||grad_i||_2 for each group i.
 * @param   abs_grad_next   similar to grad_next, but for abs_grad.
 */
template <class XType, class GroupsType,
          class GroupSizesType, class ValueType, class PenaltyType,
          class LmdasType, class BetasType,
          class ResidsType, class ISType, class GradType>
ADELIE_CORE_STRONG_INLINE 
auto check_kkt(
    const XType& X, 
    const GroupsType& groups,
    const GroupSizesType& group_sizes,
    ValueType alpha,
    const PenaltyType& penalty,
    const LmdasType& lmdas, 
    const BetasType& betas,
    const ResidsType& resids,
    const ISType& is_strong,
    size_t n_threads,
    GradType& grad,
    GradType& grad_next,
    GradType& abs_grad,
    GradType& abs_grad_next
)
{
    assert(X.cols() == grad.size());
    assert(grad.size() == grad_next.size());
    assert(resids.cols() == lmdas.size());
    assert(resids.rows() == X.rows());
    assert(abs_grad.size() == groups.size());
    assert(abs_grad.size() == abs_grad_next.size());

    size_t i = 0;
    auto alpha_c = 1 - alpha;

    for (; i < lmdas.size(); ++i) {
        const auto& beta_i = betas[i];
        const auto resid_i = resids.col(i);
        const auto lmda = lmdas[i];

        // check KKT
        std::atomic<bool> kkt_fail(false);

#pragma omp parallel for schedule(auto) num_threads(n_threads)
        for (size_t k = 0; k < groups.size(); ++k) {            
            const auto gk = groups[k];
            const auto gk_size = group_sizes[k];
            const auto X_k = X.block(0, gk, X.rows(), gk_size);
            auto grad_k = grad_next.segment(gk, gk_size);

            // Just omit the KKT check for strong variables.
            // If KKT failed previously, just do a no-op until loop finishes.
            // This is because OpenMP doesn't allow break statements.
            bool kkt_fail_raw = kkt_fail.load(std::memory_order_relaxed);
            if (kkt_fail_raw) continue;
            
            const auto pk = penalty[k];
            grad_k.noalias() = X_k.transpose() * resid_i;
            grad_k -= (lmda * alpha_c * pk) * beta_i.segment(gk, gk_size);
            const auto abs_grad_k = grad_k.norm();
            abs_grad_next[k] = abs_grad_k;
            if (is_strong(k) || (abs_grad_k <= lmda * alpha * pk)) continue;
            kkt_fail.store(true, std::memory_order_relaxed);
        }

        if (kkt_fail.load(std::memory_order_relaxed)) break; 
        else {
            grad.swap(grad_next);
            abs_grad.swap(abs_grad_next);
        }
    }

    return i;
}

    ADELIE_CORE_STRONG_INLINE
    void screen(
        size_t current_size,
        size_t delta_strong_size,
        bool do_strong_rule,
        bool all_lambdas_failed,
        bool some_lambdas_failed,
        value_t lmda_prev_valid,
        value_t lmda_next,
        bool is_lmda_prev_valid_max,
        size_t n_threads
    )
    {
        // update residual to previous valid
        // NOTE: MUST come first before screen_edpp!!
        resid = resid_prev_valid;

        // screen to append to strong set and strong hashset.
        // Only screen if KKT failure happened somewhere in the current lambda vector.
        // Otherwise, the current set might be enough for the next lambdas, so we try the current list first.

        // Note: DO NOT UPDATE strong_order YET!
        // Updating previously valid beta requires the old order.

        // updates on these will be done later!
        strong_beta_prev_valid.resize(strong_beta.size(), 0);

        // At this point, strong_set is ordered for all old variables
        // and unordered for the last few (new variables).
        // But all referencing quantities (strong_beta, strong_grad, strong_is_active, strong_vars)
        // match up in size and positions with strong_set.
        // Note: strong_grad has been updated properly to previous valid version in all cases.

        // create dense viewers of old strong betas
        Eigen::Map<vec_value_t> old_strong_beta_view(
                strong_beta.data(), old_total_strong_size);
        Eigen::Map<vec_value_t> strong_beta_prev_valid_view(
                strong_beta_prev_valid.data(),
                old_total_strong_size);

        // Save last valid strong beta and put current state to that point.
        if (all_lambdas_failed) {
            // warm-start using previously valid solution.
            // Note: this is crucial for correctness in grad update step below.
            old_strong_beta_view = strong_beta_prev_valid_view;
        }
        else if (some_lambdas_failed) {
            // save last valid solution (this logic assumes the ordering is in old one)
            assert(betas.size() > 0);
            const auto& last_valid_sol = betas.back();
            Eigen::Map<const vec_index_t> ss_map(
                strong_set.data(), old_strong_set_size
            );
            Eigen::Map<const vec_index_t> so_map(
                strong_order.data(), old_strong_set_size
            );
            Eigen::Map<const vec_index_t> sb_map(
                strong_begins.data(), old_strong_set_size
            );
            last_valid_strong_beta(
                old_strong_beta_view, last_valid_sol,
                groups, group_sizes, ss_map, so_map, sb_map
            );
            strong_beta_prev_valid_view = old_strong_beta_view;
        } else {
            strong_beta_prev_valid_view = old_strong_beta_view;
        }

        // save last valid R^2
        // TODO: with the new change to initial fitting, we can just take rsqs.back().
        assert(rsqs.size());
        rsq_prev_valid = rsqs.back();
    }
};    


{
    // first index of KKT failure
    size_t idx = 1;         
    
    const auto tidy_up = [&]() {
        // TODO: not sure if the arguments are correct in screen.
        const auto last_lmda = (lmdas.size() == 0) ? std::numeric_limits<value_t>::max() : lmdas.back();
        // Last screening to ensure that the basil state is at the previously valid state.
        // We force non-strong rule and add 0 new variables to preserve the state.
        basil_state.screen(
            0, 0, false, true, true,
            last_lmda, last_lmda, n_lambdas_rem == max_n_lambdas, n_threads
        );

        betas_out = std::move(betas);
        rsqs_out = std::move(rsqs);
        checkpoint = std::move(basil_state);
    };

    while (1) 
    {
        /* Update lambda sequence */
        const bool some_lambdas_failed = idx < lmdas_curr.size();

        const auto& n_lmdas = fit_state.n_lmdas;

        diagnostic.time_kkt.push_back(0);
        {
            sw_t stopwatch(diagnostic.time_kkt.back());
            idx = check_kkt(
                [&](auto i) { return basil_state.skip_screen(i); }, 
                n_threads, grad, grad_next, abs_grad, abs_grad_next
            );
            if (idx) {
                resid_prev_valid = resids_curr.col(idx-1);
            }
        }
    }

    tidy_up();
}

} // namespace solver
} // namespace adelie_core