#pragma once
#include <adelie_core/optimization/search_pivot.hpp>
#include <adelie_core/solver/utils.hpp>
#include <adelie_core/util/algorithm.hpp>
#include <adelie_core/util/exceptions.hpp>
#include <adelie_core/util/macros.hpp>
#include <adelie_core/util/stopwatch.hpp>
#include <adelie_core/util/tqdm.hpp>

namespace adelie_core {
namespace solver {

template <class StateType, class PBType>
ADELIE_CORE_STRONG_INLINE
void pb_add_suffix(
    const StateType& state,
    PBType& pb
)
{
    const auto& devs = state.devs;
    // current training % dev explained
    pb << " [dev:" 
        << std::fixed << std::setprecision(1) 
        << ((devs.size() == 0) ? 0.0 : devs.back()) * 100
        << "%]"
        ; 
}

template <class StateType>
ADELIE_CORE_STRONG_INLINE
bool early_exit(
    const StateType& state
)
{
    const auto early_exit = state.early_exit;
    const auto& devs = state.devs;

    if (!early_exit || !devs.size()) return false;

    const auto adev_tol = state.adev_tol;
    const auto ddev_tol = state.ddev_tol;

    const auto dev_u = devs[devs.size()-1];
    if (dev_u >= adev_tol) return true;
    if (devs.size() == 1) return false;

    const auto dev_m = devs[devs.size()-2];
    if (std::abs(dev_u-dev_m) < ddev_tol) return true;

    return false;
}

/**
 * Screens for new variables to include in the screen set
 * for fitting with lmda = lmda_next.
 * 
 * State MUST be a valid state satisfying its invariance.
 * Note that only the screen set is modified!
 * All derived screen quantities must be updated afterwards. 
 */
template <class StateType, class ValueType>
ADELIE_CORE_STRONG_INLINE 
void screen(
    StateType& state,
    ValueType lmda_next,
    bool all_kkt_passed,
    int n_new_active
)
{
    using state_t = std::decay_t<StateType>;
    using vec_value_t = typename state_t::vec_value_t;
    using vec_index_t = typename state_t::vec_index_t;

    const auto& abs_grad = state.abs_grad;
    const auto lmda = state.lmda;
    const auto alpha = state.alpha;
    const auto& penalty = state.penalty;
    const auto& screen_hashset = state.screen_hashset;
    const auto max_screen_size = state.max_screen_size;
    const auto screen_rule = state.screen_rule;
    const auto pivot_subset_ratio = state.pivot_subset_ratio;
    const auto pivot_subset_min = state.pivot_subset_min;
    const auto pivot_slack_ratio = state.pivot_slack_ratio;

    // may get modified
    auto& screen_set = state.screen_set;

    const int old_screen_set_size = screen_set.size();

    assert(screen_set.size() <= abs_grad.size());

    const auto is_screen = [&](auto i) { 
        return screen_hashset.find(i) != screen_hashset.end(); 
    };

    const auto do_pivot = [&]() {
        const int G = abs_grad.size();
        vec_index_t order = vec_index_t::LinSpaced(G, 0, G-1);
        vec_value_t weights = vec_value_t::NullaryExpr(
            G, [&](auto i) { 
                return (penalty[i] <= 0) ? 
                    alpha * lmda : std::min(abs_grad[i] / penalty[i], alpha * lmda); 
            }
        );
        std::sort(
            order.data(), 
            order.data() + order.size(), 
            [&](auto i, auto j) { 
                return weights[i] < weights[j]; 
            }
        );
        const int subset_size = std::min<int>(std::max<int>(
            old_screen_set_size * (1 + pivot_subset_ratio),
            pivot_subset_min
        ), G);
        // top largest subset_size number of weights
        vec_value_t weights_sorted_sub = vec_value_t::NullaryExpr(
            subset_size,
            [&](auto i) { return weights[order[G-subset_size+i]]; } 
        );

        vec_value_t mses(subset_size);
        vec_value_t indices = vec_value_t::LinSpaced(subset_size, 0, subset_size-1);
        const int pivot_idx = optimization::search_pivot(
            indices, 
            weights_sorted_sub, 
            mses
        );
        const int full_pivot_idx = G - subset_size + pivot_idx;

        // add everything beyond the cutoff index that isn't screen yet
        for (int ii = G-1; ii >= full_pivot_idx; --ii) {
            const auto i = order[ii];
            if (is_screen(i)) continue;
            screen_set.push_back(i); 
        }
        // add some slack of new groups below the pivot
        int count = 0;
        for (int ii = full_pivot_idx - 1; ii >= 0; --ii) {
            if (count >= pivot_slack_ratio * std::max<int>(n_new_active, 1)) break;
            const auto i = order[ii]; 
            if (is_screen(i)) continue;
            screen_set.push_back(i);
            ++count;
        }

        // this case should rarely happen, but we arrived here because
        // previous iteration added all pivot-rule predictions and KKT still failed.
        // In this case, do the most safe thing, which is to add all failed variables.
        if ((screen_set.size() == static_cast<size_t>(old_screen_set_size)) && !all_kkt_passed) {
            for (int i = 0; i < abs_grad.size(); ++i) {
                if (is_screen(i)) continue;
                if (abs_grad[i] > lmda_next * penalty[i] * alpha) {
                    screen_set.push_back(i);
                }
            }
        }
    };

    /* update screen_set */

    // KKT passed for some lambdas in the batch
    if (screen_rule == util::screen_rule_type::_strong) {
        const auto strong_rule_lmda = (2 * lmda_next - lmda) * alpha;

        for (int i = 0; i < abs_grad.size(); ++i) {
            if (is_screen(i)) continue;
            if (abs_grad[i] > strong_rule_lmda * penalty[i]) {
                screen_set.push_back(i);
            }
        }
    } else if (screen_rule == util::screen_rule_type::_pivot) {
        do_pivot();
    } else {
        throw std::runtime_error("Unknown screen rule!");
    }

    // If adding new amount went over max screen size, 
    // undo the change to keep invariance from before, then throw exception.
    if (screen_set.size() > max_screen_size) {
        screen_set.erase(
            std::next(screen_set.begin(), old_screen_set_size),
            screen_set.end()
        );
        throw util::max_basil_screen_set();
    }
}

/**
 * Checks the KKT condition for the current state at lmda.
 */
template <class StateType, class ValueType>
ADELIE_CORE_STRONG_INLINE
bool kkt(
    StateType& state,
    ValueType lmda
)
{
    const auto& groups = state.groups;
    const auto alpha = state.alpha;
    const auto& penalty = state.penalty;
    const auto& screen_hashset = state.screen_hashset;
    const auto& abs_grad = state.abs_grad;

    const auto is_screen = [&](auto i) {
        return screen_hashset.find(i) != screen_hashset.end();
    };

    for (int k = 0; k < groups.size(); ++k) {
        if (is_screen(k)) continue;
        const auto pk = penalty[k];
        const auto abs_grad_k = abs_grad[k];
        if (abs_grad_k > lmda * alpha * pk) return false;
    }

    return true;
}

template <class StateType,
          class PBAddSuffixType,
          class UpdateLossNullType,
          class UpdateInvarianceType,
          class UpdateSolutionsType,
          class EarlyExitType,
          class ScreenType,
          class FitType>
inline void solve_core(
    StateType&& state,
    bool display,
    PBAddSuffixType pb_add_suffix_f,
    UpdateLossNullType update_loss_null_f,
    UpdateInvarianceType update_invariance_f,
    UpdateSolutionsType update_solutions_f,
    EarlyExitType early_exit_f,
    ScreenType screen_f,
    FitType fit_f
)
{
    using state_t = std::decay_t<StateType>;
    using value_t = typename state_t::value_t;
    using vec_value_t = typename state_t::vec_value_t;
    using vec_safe_bool_t = typename state_t::vec_safe_bool_t;
    using sw_t = util::Stopwatch;

    const auto alpha = state.alpha;
    const auto& penalty = state.penalty;
    const auto& screen_set = state.screen_set;
    const auto max_screen_size = state.max_screen_size;
    const auto setup_lmda_max = state.setup_lmda_max;
    const auto setup_lmda_path = state.setup_lmda_path;
    const auto lmda_path_size = state.lmda_path_size;
    const auto min_ratio = state.min_ratio;
    const auto& screen_is_active = state.screen_is_active;
    const auto& abs_grad = state.abs_grad;
    auto& lmda_max = state.lmda_max;
    auto& lmda_path = state.lmda_path;
    auto& benchmark_screen = state.benchmark_screen;
    auto& benchmark_fit_screen = state.benchmark_fit_screen;
    auto& benchmark_fit_active = state.benchmark_fit_active;
    auto& benchmark_kkt = state.benchmark_kkt;
    auto& benchmark_invariance = state.benchmark_invariance;
    auto& n_valid_solutions = state.n_valid_solutions;
    auto& active_sizes = state.active_sizes;
    auto& screen_sizes = state.screen_sizes;

    if (screen_set.size() > max_screen_size) throw util::max_basil_screen_set();

    update_loss_null_f(state);

    // ==================================================================================== 
    // Initial fit for lambda ~ infinity to setup lmda_max
    // ==================================================================================== 
    // Only unpenalized (l1) groups are active in this case.
    // State must include all unpenalized groups.
    // We solve for large lambda, then back-track the KKT condition to find the lambda
    // that leads to that solution where all penalized variables have 0 coefficient.
    if (setup_lmda_max) {
        const auto large_lmda = std::numeric_limits<value_t>::max(); 

        fit_f(state, large_lmda);

        update_invariance_f(state, large_lmda);

        lmda_max = compute_lmda_max(abs_grad, alpha, penalty);
    }

    // ==================================================================================== 
    // Generate lambda path if needed
    // ==================================================================================== 
    if (setup_lmda_path) {
        if (lmda_path_size <= 0) throw std::runtime_error("lmda_path_size must be > 0.");

        lmda_path.resize(lmda_path_size);

        compute_lmda_path(lmda_path, min_ratio, lmda_max);
    }

    // ==================================================================================== 
    // Initial fit for lambda > lambda_max
    // ==================================================================================== 
    // Only unpenalized (l1) groups are active in this case by definition of lmda_max.
    // Since state is in its invariance (solution at state.lmda) and unpenalized groups
    // are ALWAYS active, state includes all unpenalized groups.
    // If no lambda in lmda_path is > lmda_max and setup at lmda_max is not required, 
    // state is left unchanged.
    // Otherwise, it is in its invariance at lmda = lmda_max.
    // All solutions to lambda > lambda_max are saved.

    // slice lambda_path up to lmda_max
    const auto large_lmda_path_size = std::find_if(
        lmda_path.data(), 
        lmda_path.data() + lmda_path.size(),
        [&](auto x) { return x <= lmda_max; }
    ) - lmda_path.data();

    if (large_lmda_path_size || setup_lmda_max) {
        // create a lambda path containing only lmdas > lambda_max
        // and additionally lambda_max at the end.
        // If large_lmda_path_size > 0, mind as well fit for lambda_max as well to go down the path.
        vec_value_t large_lmda_path(large_lmda_path_size + 1);
        large_lmda_path.head(large_lmda_path_size) = lmda_path.head(large_lmda_path_size);
        large_lmda_path[large_lmda_path_size] = lmda_max;

        for (int i = 0; i < large_lmda_path.size(); ++i) {
            auto tup = fit_f(state, large_lmda_path[i]);
            auto&& state_gaussian_pin = std::get<0>(tup);

            /* Invariance */
            // save only the solutions that the user asked for (up to and not including lmda_max)
            if (i < large_lmda_path.size()-1) {
                update_solutions_f(
                    state, 
                    state_gaussian_pin,
                    large_lmda_path[i]
                );
            // otherwise, put the state at the last fitted lambda (lmda_max)
            } else {
                update_invariance_f(state, large_lmda_path[i]);
            }
        }
    }

    size_t lmda_path_idx = large_lmda_path_size; // next index into lmda_path to fit

    // ==================================================================================== 
    // BASIL iterations for lambda <= lambda_max
    // ==================================================================================== 
    // In this case, screen_set may not contain the true active set.
    // We must go through BASIL iterations to solve each lambda.
    sw_t sw;
    int current_active_size = Eigen::Map<const vec_safe_bool_t>(
        screen_is_active.data(),
        screen_is_active.size()
    ).sum();
    bool kkt_passed = true;
    int n_new_active = 0;

    auto pb = util::tq::trange(lmda_path.size() - lmda_path_idx);
    pb.set_display(display);

    for (int _ : pb)
    {
        static_cast<void>(_);

        // check early exit
        if (early_exit_f(state)) {
            pb_add_suffix_f(state, pb);
            break;
        }

        // batch the next set of lambdas
        const auto lmda_curr = lmda_path[lmda_path_idx];

        // keep doing screen-fit-kkt until KKT passes
        while (1) {
            try {
                // ==================================================================================== 
                // Screening step
                // ==================================================================================== 
                sw.start();
                screen_f(state, lmda_curr, kkt_passed, n_new_active);
                benchmark_screen.push_back(sw.elapsed());

                // ==================================================================================== 
                // Fit step
                // ==================================================================================== 
                auto tup = fit_f(state, lmda_curr);
                auto&& state_gaussian_pin = std::get<0>(tup);
                benchmark_fit_screen.push_back(std::get<1>(tup));
                benchmark_fit_active.push_back(std::get<2>(tup));

                // ==================================================================================== 
                // Invariance step
                // ==================================================================================== 
                sw.start();
                update_invariance_f(state, lmda_curr);
                benchmark_invariance.push_back(sw.elapsed());

                // ==================================================================================== 
                // KKT step
                // ==================================================================================== 
                sw.start();
                kkt_passed = kkt(state, lmda_curr);
                n_valid_solutions.push_back(kkt_passed);
                lmda_path_idx += kkt_passed;
                if (kkt_passed) {
                    update_solutions_f(
                        state, 
                        state_gaussian_pin,
                        lmda_curr
                    );
                }
                benchmark_kkt.push_back(sw.elapsed());

                // ==================================================================================== 
                // Diagnostic step
                // ==================================================================================== 
                if (kkt_passed) {
                    active_sizes.push_back(state_gaussian_pin.active_set.size());
                    screen_sizes.push_back(state.screen_set.size());
                }
                // compute the number of new active groups 
                n_new_active = (
                    kkt_passed ?
                    active_sizes.back() - current_active_size : n_new_active
                );
                current_active_size = (
                    kkt_passed ?
                    active_sizes.back() : current_active_size
                );
            } catch (...) {
                pb_add_suffix_f(state, pb);
                throw;
            }

            if (kkt_passed) break;
        } // end while(1)

        pb_add_suffix_f(state, pb);
    }
}

} // namespace solver
} // namespace adelie_core 