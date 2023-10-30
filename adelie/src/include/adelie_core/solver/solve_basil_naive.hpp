#pragma once
#include <adelie_core/matrix/matrix_naive_base.hpp>
#include <adelie_core/optimization/search_pivot.hpp>
#include <adelie_core/solver/solve_basil_base.hpp>
#include <adelie_core/solver/solve_pin_naive.hpp>
#include <adelie_core/state/state_basil_naive.hpp>
#include <adelie_core/state/state_pin_naive.hpp>
#include <adelie_core/util/algorithm.hpp>
#include <adelie_core/util/stopwatch.hpp>

namespace adelie_core {
namespace solver {
namespace naive {

template <class StateType, class StatePinType>
ADELIE_CORE_STRONG_INLINE
void update_solutions(
    StateType& state,
    StatePinType& state_pin_naive,
    size_t n_sols
)
{
    const auto y_var = state.y_var;
    auto& betas = state.betas;
    auto& rsqs = state.rsqs;
    auto& lmdas = state.lmdas;
    auto& intercepts = state.intercepts;

    for (size_t i = 0; i < n_sols; ++i) {
        betas.emplace_back(std::move(state_pin_naive.betas[i]));
        intercepts.emplace_back(state_pin_naive.intercepts[i]);
        rsqs.emplace_back(state_pin_naive.rsqs[i]);
        lmdas.emplace_back(state_pin_naive.lmdas[i]);

        // normalize R^2 by null model
        rsqs.back() /= y_var;
    }
}

/**
 * Screens for new variables to include in the safe set
 * for fitting with lmda = lmda_next.
 * 
 * State MUST be a valid state satisfying its invariance.
 * The state after the function is finished is also a valid state.
 */
template <class StateType, class ValueType>
ADELIE_CORE_STRONG_INLINE
void screen_edpp(
    StateType& state,
    ValueType lmda_next
)
{
    using state_t = std::decay_t<StateType>;
    using vec_value_t = typename state_t::vec_value_t;

    const auto use_edpp = state.use_edpp;
    const auto& groups = state.groups;
    const auto& X_means = state.X_means;
    const auto& X_group_norms = state.X_group_norms;
    const auto& group_sizes = state.group_sizes;
    const auto& penalty = state.penalty;
    const auto lmda = state.lmda;
    const auto lmda_max = state.lmda_max;
    const auto& edpp_v1_0 = state.edpp_v1_0;
    const auto& edpp_resid_0 = state.edpp_resid_0;
    const auto& strong_X_means = state.strong_X_means;
    const auto& strong_beta = state.strong_beta;
    const auto& resid = state.resid;
    const auto& grad = state.grad;
    const auto intercept = state.intercept;
    auto& X = *state.X;
    auto& edpp_safe_set = state.edpp_safe_set;
    auto& edpp_safe_hashset = state.edpp_safe_hashset;

    if (!use_edpp || (groups.size() == edpp_safe_hashset.size())) return;

    vec_value_t v1 = (
        (lmda == lmda_max) ?
        edpp_v1_0 :
        (edpp_resid_0 - resid) / lmda
    );
    vec_value_t v2 = edpp_resid_0 / lmda_next - resid / lmda;
    if (intercept) {
        const auto resid_correction = (
            Eigen::Map<const vec_value_t>(strong_X_means.data(), strong_X_means.size()) * 
            Eigen::Map<const vec_value_t>(strong_beta.data(), strong_beta.size())
        ).sum() / lmda;
        if (lmda != lmda_max) v1 -= resid_correction;    
        v2 -= resid_correction;
    }
    vec_value_t v2_perp = v2 - (v1.matrix().dot(v2.matrix()) / v1.matrix().squaredNorm()) * v1;
    const auto v2_perp_norm = v2_perp.matrix().norm();

    vec_value_t buffer(grad.size());    

    for (int i = 0; i < groups.size(); ++i) {
        if (edpp_safe_hashset.find(i) != edpp_safe_hashset.end()) continue;
        const auto g = groups[i];
        const auto gs = group_sizes[i];
        const auto grad_g = grad.segment(g, gs);
        auto buff_g = buffer.segment(g, gs);

        X.bmul(g, gs, v2_perp, buff_g);
        buff_g = 0.5 * buff_g + grad_g / lmda;
        if (intercept) {
            buff_g -= (0.5 * v2_perp.sum()) * X_means.segment(g, gs);
        }
        const auto buff_g_norm = buff_g.matrix().norm();

        if (buff_g_norm >= penalty[i] - 0.5 * v2_perp_norm * X_group_norms[i]) {
            edpp_safe_set.push_back(i);
            edpp_safe_hashset.emplace(i);
        }
    }    
}

/**
 * Screens for new variables to include in the strong set
 * for fitting with lmda = lmda_next.
 * 
 * State MUST be a valid state satisfying its invariance.
 * The state after the function is finished is also a valid state.
 */
template <class StateType, class ValueType>
ADELIE_CORE_STRONG_INLINE 
void screen_strong(
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
    const auto& strong_hashset = state.strong_hashset;
    const auto& edpp_safe_set = state.edpp_safe_set;
    const auto& edpp_safe_hashset = state.edpp_safe_hashset;
    const auto max_strong_size = state.max_strong_size;
    const auto screen_rule = state.screen_rule;
    const auto lazify_screen = state.lazify_screen;
    const auto pivot_subset_ratio = state.pivot_subset_ratio;
    const auto pivot_subset_min = state.pivot_subset_min;
    const auto pivot_slack_ratio = state.pivot_slack_ratio;

    // may get modified
    auto delta_strong_size = state.delta_strong_size;
    auto& strong_set = state.strong_set;

    const int old_strong_set_size = strong_set.size();
    const int new_safe_size = edpp_safe_set.size() - old_strong_set_size;

    assert(strong_set.size() <= abs_grad.size());

    const auto is_strong = [&](auto i) { 
        return strong_hashset.find(i) != strong_hashset.end(); 
    };
    const auto is_edpp = [&](auto i) { 
        return edpp_safe_hashset.find(i) != edpp_safe_hashset.end(); 
    };

    const auto do_fixed_greedy = [&]() {
        size_t size_capped = std::min<int>(
            delta_strong_size, 
            new_safe_size
        );

        strong_set.insert(strong_set.end(), size_capped, -1);
        const auto abs_grad_p = vec_value_t::NullaryExpr(
            abs_grad.size(), [&](auto i) {
                return (penalty[i] <= 0 || !is_edpp(i)) ? 0.0 : abs_grad[i] / penalty[i];
            }
        );
        util::k_imax(
            abs_grad_p, 
            is_strong,
            size_capped, 
            std::next(strong_set.begin(), old_strong_set_size)
        );
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
            old_strong_set_size * (1 + pivot_subset_ratio),
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

        // add everything beyond the cutoff index that isn't strong yet
        for (int ii = G-1; ii >= full_pivot_idx; --ii) {
            const auto i = order[ii];
            if (is_strong(i) || !is_edpp(i)) continue;
            strong_set.push_back(i); 
        }
        // add some slack of new groups below the pivot
        int count = 0;
        for (int ii = full_pivot_idx - 1; ii >= 0; --ii) {
            if (count >= pivot_slack_ratio * (n_new_active + 1)) break;
            const auto i = order[ii]; 
            if (is_strong(i) || !is_edpp(i)) continue;
            strong_set.push_back(i);
            ++count;
        }

        // this case should rarely happen, but we arrived here because
        // previous iteration added all pivot-rule predictions and KKT still failed.
        // In this case, do the most safe thing, which is to add all failed variables.
        if ((strong_set.size() == old_strong_set_size) && !all_kkt_passed) {
            for (int i = 0; i < abs_grad.size(); ++i) {
                if (is_strong(i) || !is_edpp(i)) continue;
                if (abs_grad[i] > lmda_next * penalty[i] * alpha) {
                    strong_set.push_back(i);
                }
            }
        }
    };

    /* update strong_set */

    // KKT passed for some lambdas in the batch
    if (lazify_screen && all_kkt_passed) {
        // only add n_new_active number of groups
        delta_strong_size = n_new_active;
        do_fixed_greedy();
    } else {
        if (screen_rule == state::screen_rule_type::_strong) {
            const auto strong_rule_lmda = (2 * lmda_next - lmda) * alpha;

            for (int i = 0; i < abs_grad.size(); ++i) {
                if (is_strong(i) || !is_edpp(i)) continue;
                if (abs_grad[i] > strong_rule_lmda * penalty[i]) {
                    strong_set.push_back(i);
                }
            }
        } else if (screen_rule == state::screen_rule_type::_fixed_greedy) {
            do_fixed_greedy();
        } else if (screen_rule == state::screen_rule_type::_safe) {
            for (int i = 0; i < edpp_safe_set.size(); ++i) {
                if (is_strong(edpp_safe_set[i])) continue;
                strong_set.push_back(edpp_safe_set[i]);
            }
        } else if (screen_rule == state::screen_rule_type::_pivot) {
            do_pivot();
        } else {
            throw std::runtime_error("Unknown strong rule!");
        }
    }

    // If adding new amount went over max strong size, 
    // undo the change to keep invariance from before, then throw exception.
    if (strong_set.size() > max_strong_size) {
        strong_set.erase(
            std::next(strong_set.begin(), old_strong_set_size),
            strong_set.end()
        );
        throw util::max_basil_strong_set();
    }

    /* update derived strong quantities */
    state::update_strong_derived_naive(state);

}

template <class StateType, class ValueType>
ADELIE_CORE_STRONG_INLINE
void screen(
    StateType& state,
    ValueType lmda_next,
    bool all_kkt_passed,
    int n_new_active
)
{
    screen_edpp(state, lmda_next);
    screen_strong(state, lmda_next, all_kkt_passed, n_new_active);
}

template <class StateType,
          class LmdaPathType,
          class UpdateCoefficientsType,
          class CUIType=util::no_op>
ADELIE_CORE_STRONG_INLINE
auto fit(
    StateType& state,
    const LmdaPathType& lmda_path,
    UpdateCoefficientsType update_coefficients_f,
    CUIType check_user_interrupt = CUIType()
)
{
    using state_t = std::decay_t<StateType>;
    using index_t = typename state_t::index_t;
    using safe_bool_t = typename state_t::safe_bool_t;
    using vec_value_t = typename state_t::vec_value_t;
    using vec_index_t = typename state_t::vec_index_t;
    using vec_safe_bool_t = util::rowvec_type<safe_bool_t>;
    using matrix_pin_naive_t = typename state_t::matrix_t;
    using state_pin_naive_t = state::StatePinNaive<
        matrix_pin_naive_t,
        typename std::decay_t<matrix_pin_naive_t>::value_t,
        index_t,
        safe_bool_t
    >;

    auto& X = *state.X;
    const auto y_mean = state.y_mean;
    const auto y_var = state.y_var;
    const auto& groups = state.groups;
    const auto& group_sizes = state.group_sizes;
    const auto alpha = state.alpha;
    const auto& penalty = state.penalty;
    const auto& strong_set = state.strong_set;
    const auto& strong_g1 = state.strong_g1;
    const auto& strong_g2 = state.strong_g2;
    const auto& strong_begins = state.strong_begins;
    const auto& strong_vars = state.strong_vars;
    const auto& strong_X_means = state.strong_X_means;
    const auto& strong_transforms = state.strong_transforms;
    const auto intercept = state.intercept;
    const auto max_iters = state.max_iters;
    const auto tol = state.tol;
    const auto rsq_tol = state.rsq_tol;
    const auto rsq_slope_tol = state.rsq_slope_tol;
    const auto rsq_curv_tol = state.rsq_curv_tol;
    const auto newton_tol = state.newton_tol;
    const auto newton_max_iters = state.newton_max_iters;
    const auto n_threads = state.n_threads;
    const auto rsq = state.rsq;
    const auto resid_sum = state.resid_sum;
    auto& resid = state.resid;
    auto& strong_beta = state.strong_beta;
    auto& strong_is_active = state.strong_is_active;

    state_pin_naive_t state_pin_naive(
        X,
        y_mean,
        y_var,
        groups, 
        group_sizes,
        alpha, 
        penalty,
        Eigen::Map<const vec_index_t>(strong_set.data(), strong_set.size()), 
        Eigen::Map<const vec_index_t>(strong_g1.data(), strong_g1.size()), 
        Eigen::Map<const vec_index_t>(strong_g2.data(), strong_g2.size()), 
        Eigen::Map<const vec_index_t>(strong_begins.data(), strong_begins.size()), 
        Eigen::Map<const vec_value_t>(strong_vars.data(), strong_vars.size()), 
        Eigen::Map<const vec_value_t>(strong_X_means.data(), strong_X_means.size()), 
        strong_transforms,
        lmda_path,
        intercept, max_iters, tol, rsq_tol, rsq_slope_tol, rsq_curv_tol, 
        newton_tol, newton_max_iters, n_threads,
        rsq,
        Eigen::Map<vec_value_t>(resid.data(), resid.size()),
        resid_sum,
        Eigen::Map<vec_value_t>(strong_beta.data(), strong_beta.size()), 
        Eigen::Map<vec_safe_bool_t>(strong_is_active.data(), strong_is_active.size())
    );

    solve_pin(
        state_pin_naive, 
        update_coefficients_f, 
        check_user_interrupt
    );

    return state_pin_naive;
}

/**
 * Checks the KKT condition on the proposed solutions in state_pin_naive.
 */
template <class StateType, class StatePinNaiveType>
ADELIE_CORE_STRONG_INLINE
size_t kkt(
    StateType& state,
    const StatePinNaiveType& state_pin_naive
)
{
    const auto& X_means = state.X_means;
    const auto& groups = state.groups;
    const auto& group_sizes = state.group_sizes;
    const auto alpha = state.alpha;
    const auto& penalty = state.penalty;
    const auto intercept = state.intercept;
    const auto n_threads = state.n_threads;
    const auto& strong_hashset = state.strong_hashset;
    const auto& edpp_safe_hashset = state.edpp_safe_hashset;
    const auto& abs_grad = state.abs_grad;
    auto& X = *state.X;
    auto& grad = state.grad;

    const auto is_strong = [&](auto i) {
        return strong_hashset.find(i) != strong_hashset.end();
    };
    const auto is_edpp_safe = [&](auto i) {
        return edpp_safe_hashset.find(i) != edpp_safe_hashset.end();
    };

    const auto p = X.cols();

    // NOTE: no matter what, every gradient must be computed for pivot method.
    X.bmul(0, p, state_pin_naive.resids[0], grad);
    if (intercept) {
        matrix::dvsubi(grad, state_pin_naive.resid_sums[0] * X_means, n_threads);
    }
    state::update_abs_grad(state);

    // First, loop over non-strong set, compute gradients, and update n_valid_solutions.
    for (int k = 0; k < groups.size(); ++k) {
        if (is_strong(k) || !is_edpp_safe(k)) continue;
        const auto pk = penalty[k];
        const auto abs_grad_k = abs_grad[k];
        const auto lmda = state_pin_naive.lmdas[0];
        if (abs_grad_k > lmda * alpha * pk) return false;
    }

    return true;
}

template <class StateType,
          class UpdateCoefficientsType,
          class CUIType=util::no_op>
inline void solve_basil(
    StateType&& state,
    UpdateCoefficientsType update_coefficients_f,
    CUIType check_user_interrupt = CUIType()
)
{
    using state_t = std::decay_t<StateType>;
    using value_t = typename state_t::value_t;
    using vec_value_t = typename state_t::vec_value_t;
    using vec_safe_bool_t = typename state_t::vec_safe_bool_t;
    using sw_t = util::Stopwatch;

    const auto& X_means = state.X_means;
    const auto alpha = state.alpha;
    const auto& penalty = state.penalty;
    const auto& strong_set = state.strong_set;
    const auto& rsqs = state.rsqs;
    const auto early_exit = state.early_exit;
    const auto max_strong_size = state.max_strong_size;
    const auto rsq_tol = state.rsq_tol;
    const auto rsq_slope_tol = state.rsq_slope_tol;
    const auto rsq_curv_tol = state.rsq_curv_tol;
    const auto setup_edpp = state.setup_edpp;
    const auto setup_lmda_max = state.setup_lmda_max;
    const auto setup_lmda_path = state.setup_lmda_path;
    const auto lmda_path_size = state.lmda_path_size;
    const auto min_ratio = state.min_ratio;
    const auto intercept = state.intercept;
    const auto n_threads = state.n_threads;
    const auto& abs_grad = state.abs_grad;
    auto& X = *state.X;
    auto& lmda_max = state.lmda_max;
    auto& lmda_path = state.lmda_path;
    auto& strong_is_active = state.strong_is_active;
    auto& strong_beta = state.strong_beta;
    auto& resid = state.resid;
    auto& resid_sum = state.resid_sum;
    auto& grad = state.grad;
    auto& rsq = state.rsq;
    auto& lmda = state.lmda;
    auto& resid_prev_valid = state.resid_prev_valid;
    auto& strong_beta_prev_valid = state.strong_beta_prev_valid; 
    auto& strong_is_active_prev_valid = state.strong_is_active_prev_valid;
    auto& benchmark_screen = state.benchmark_screen;
    auto& benchmark_fit_strong = state.benchmark_fit_strong;
    auto& benchmark_fit_active = state.benchmark_fit_active;
    auto& benchmark_kkt = state.benchmark_kkt;
    auto& benchmark_invariance = state.benchmark_invariance;
    auto& n_valid_solutions = state.n_valid_solutions;
    auto& active_sizes = state.active_sizes;
    auto& strong_sizes = state.strong_sizes;
    auto& edpp_safe_sizes = state.edpp_safe_sizes;

    const auto p = grad.size();

    if (strong_set.size() > max_strong_size) throw util::max_basil_strong_set();

    const auto save_prev_valid = [&]() {
        resid_prev_valid = resid;
        strong_beta_prev_valid = strong_beta;
        strong_is_active_prev_valid = strong_is_active;
    };
    const auto load_prev_valid = [&]() {
        resid.swap(resid_prev_valid);
        strong_beta.swap(strong_beta_prev_valid);
        strong_is_active.swap(strong_is_active_prev_valid);
    };

    // ==================================================================================== 
    // Initial fit for lambda ~ infinity to setup lmda_max
    // ==================================================================================== 
    // Only unpenalized (l1) groups are active in this case.
    // State must include all unpenalized groups.
    // We solve for large lambda, then back-track the KKT condition to find the lambda
    // that leads to that solution where all penalized variables have 0 coefficient.
    if (setup_lmda_max) {
        vec_value_t large_lmda_path(1);
        large_lmda_path[0] = std::numeric_limits<value_t>::max();
        try {
            save_prev_valid();
            auto&& state_pin_naive = naive::fit(
                state,
                large_lmda_path,
                update_coefficients_f,
                check_user_interrupt
            );

            /* Invariance */
            resid_sum = state_pin_naive.resid_sums.back();
            rsq = state_pin_naive.rsqs.back();
            lmda = state_pin_naive.lmdas.back();
            X.bmul(0, p, resid, grad);
            if (intercept) {
                matrix::dvsubi(grad, resid_sum * X_means, n_threads);
            }
            state::update_abs_grad(state);

            /* Compute lmda_max */
            const auto factor = (alpha <= 0) ? 1e-3 : alpha;
            lmda_max = vec_value_t::NullaryExpr(
                abs_grad.size(),
                [&](auto i) { 
                    return (penalty[i] <= 0.0) ? 0.0 : abs_grad[i] / penalty[i];
                }
            ).maxCoeff() / factor;
        } catch (const std::exception& e) {
            load_prev_valid();
            throw util::propagator_error(e.what());
        }
    }

    // ==================================================================================== 
    // Generate lambda path if needed
    // ==================================================================================== 
    if (setup_lmda_path) {
        if (lmda_path_size <= 0) throw std::runtime_error("lmda_path_size must be > 0.");

        lmda_path.resize(lmda_path_size);

        if (lmda_path_size == 1) lmda_path[0] = lmda_max;

        const auto log_factor = std::log(min_ratio) / (lmda_path_size - 1);
        lmda_path = lmda_max * (log_factor * vec_value_t::LinSpaced(
            lmda_path_size, 0, lmda_path_size-1
        )).exp();
        lmda_path[0] = lmda_max; // for numerical stability
    }

    // ==================================================================================== 
    // Initial fit for lambda > lambda_max
    // ==================================================================================== 
    // Only unpenalized (l1) groups are active in this case by definition of lmda_max.
    // Since state is in its invariance (solution at state.lmda) and unpenalized groups
    // are ALWAYS active, state includes all unpenalized groups.
    // If no lambda in lmda_path is > lmda_max and EDPP setup not needed, state is left unchanged.
    // Otherwise, it is in its invariance at lmda = lmda_max.
    // All solutions to lambda > lambda_max are saved.

    // slice lambda_path up to lmda_max
    const auto large_lmda_path_size = std::find_if(
        lmda_path.data(), 
        lmda_path.data() + lmda_path.size(),
        [&](auto x) { return x <= lmda_max; }
    ) - lmda_path.data();

    // if there is some large lambda or EDPP setup is needed or lmda_max setup is needed
    // do the initial fit up to (and including) lmda_max.
    if (large_lmda_path_size || setup_edpp || setup_lmda_max) {
        // create a lambda path containing only lmdas > lambda_max
        // and additionally lambda_max at the end.
        // If large_lmda_path_size > 0, mind as well fit for lambda_max as well to go down the path.
        // Else, setup_edpp is true, in which case, we must fit lambda_max.
        vec_value_t large_lmda_path(large_lmda_path_size + 1);
        large_lmda_path.head(large_lmda_path_size) = lmda_path.head(large_lmda_path_size);
        large_lmda_path[large_lmda_path_size] = lmda_max;

        try {
            save_prev_valid();
            auto&& state_pin_naive = naive::fit(
                state, 
                large_lmda_path, 
                update_coefficients_f, 
                check_user_interrupt
            );

            /* Invariance */
            // put the state at the last fitted lambda (should be lmda_max)
            // but save only the solutions that the user asked for (up to and not including lmda_max).
            resid.swap(state_pin_naive.resids.back());
            resid_sum = state_pin_naive.resid_sums.back();
            // TODO: implement swap
            Eigen::Map<vec_value_t>(
                strong_beta.data(),
                strong_beta.size()
            ) = state_pin_naive.strong_betas.back();
            Eigen::Map<vec_safe_bool_t>(
                strong_is_active.data(), 
                strong_is_active.size()
            ) = state_pin_naive.strong_is_actives.back();
            rsq = state_pin_naive.rsqs.back();
            lmda = state_pin_naive.lmdas.back();
            X.bmul(0, p, resid, grad);
            if (intercept) {
                matrix::dvsubi(grad, resid_sum * X_means, n_threads);
            }
            state::update_abs_grad(state);
            naive::update_solutions(
                state, 
                state_pin_naive, 
                state_pin_naive.lmdas.size()-1
            );
        } catch (const std::exception& e) {
            load_prev_valid();
            throw util::propagator_error(e.what());
        }
    }

    size_t lmda_path_idx = rsqs.size(); // next index into lmda_path to fit

    // ==================================================================================== 
    // Update state after initial fit
    // ==================================================================================== 
    // The state is in its invariance and 
    // is right at the solution at lambda_max if it needed to be fit from before.
    // The latter state is needed if EDPP states need to be updated.
    // Otherwise, EDPP states are unmodified.
    state::update_edpp_states(state);

    // From this point on, all EDPP states are valid, whether they were updated or not.

    // ==================================================================================== 
    // BASIL iterations for lambda <= lambda_max
    // ==================================================================================== 
    // In this case, strong_set may not contain the true active set.
    // We must go through BASIL iterations to solve each lambda.
    vec_value_t lmda_batch;
    sw_t sw;
    int current_active_size = Eigen::Map<vec_safe_bool_t>(
        strong_is_active.data(),
        strong_is_active.size()
    ).sum();
    bool kkt_passed = true;
    int n_new_active = 0;

    while (1) 
    {
        // check early exit
        if (early_exit && (rsqs.size() >= 3)) {
            const auto rsq_u = rsqs[rsqs.size()-1];
            const auto rsq_m = rsqs[rsqs.size()-2];
            const auto rsq_l = rsqs[rsqs.size()-3];
            if (check_early_stop_rsq(rsq_l, rsq_m, rsq_u, rsq_slope_tol, rsq_curv_tol) || 
                (rsqs.back() >= rsq_tol)) break;
        }

        // check if any lambdas left to fit
        if (lmda_path_idx >= lmda_path.size()) break;

        // batch the next set of lambdas
        lmda_batch = lmda_path.segment(
            lmda_path_idx, 1
        );

        // ==================================================================================== 
        // Screening step
        // ==================================================================================== 
        sw.start();
        naive::screen(
            state,
            lmda_batch[0],
            kkt_passed,
            n_new_active
        );
        benchmark_screen.push_back(sw.elapsed());

        try {
            // ==================================================================================== 
            // Fit step
            // ==================================================================================== 
            // Save all current valid quantities that will be modified in-place by fit.
            // This is needed in case we exit with exception and need to restore invariance.
            save_prev_valid();
            auto&& state_pin_naive = naive::fit(
                state,
                lmda_batch,
                update_coefficients_f,
                check_user_interrupt
            );
            benchmark_fit_strong.push_back(
                Eigen::Map<const util::rowvec_type<double>>(
                    state_pin_naive.benchmark_strong.data(),
                    state_pin_naive.benchmark_strong.size()
                ).sum()
            );
            benchmark_fit_active.push_back(
                Eigen::Map<const util::rowvec_type<double>>(
                    state_pin_naive.benchmark_active.data(),
                    state_pin_naive.benchmark_active.size()
                ).sum()
            );

            // ==================================================================================== 
            // KKT step
            // ==================================================================================== 
            sw.start();
            kkt_passed = naive::kkt(
                state,
                state_pin_naive
            );
            benchmark_kkt.push_back(sw.elapsed());
            n_valid_solutions.push_back(kkt_passed);

            // ==================================================================================== 
            // Invariance step
            // ==================================================================================== 
            sw.start();
            lmda_path_idx += kkt_passed;
            resid_sum = state_pin_naive.resid_sums[0];
            rsq = state_pin_naive.rsqs[0];
            lmda = lmda_batch[0];
            naive::update_solutions(
                state, 
                state_pin_naive, 
                kkt_passed
            );
            benchmark_invariance.push_back(sw.elapsed());

            // ==================================================================================== 
            // Diagnostic step
            // ==================================================================================== 
            if (kkt_passed) {
                active_sizes.push_back(state_pin_naive.active_set.size());
                strong_sizes.push_back(state.strong_set.size());
                edpp_safe_sizes.push_back(state.edpp_safe_set.size());
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
        } catch (const std::exception& e) {
            load_prev_valid();
            throw util::propagator_error(e.what());
        }
    }
}

} // namespace naive
} // namespace solver
} // namespace adelie_core