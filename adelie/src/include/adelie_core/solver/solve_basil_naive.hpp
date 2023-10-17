#pragma once
#include <adelie_core/matrix/matrix_naive_base.hpp>
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

    if (!use_edpp) return;

    const auto& X_means = state.X_means;
    const auto& X_group_norms = state.X_group_norms;
    const auto& groups = state.groups;
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
    const auto n_threads = state.n_threads;
    auto& X = *state.X;
    auto& edpp_safe_set = state.edpp_safe_set;
    auto& edpp_safe_hashset = state.edpp_safe_hashset;

    const auto resid_correction = (
        Eigen::Map<const vec_value_t>(strong_X_means.data(), strong_X_means.size()) * 
        Eigen::Map<const vec_value_t>(strong_beta.data(), strong_beta.size())
    ).sum() / lmda;

    vec_value_t v1 = (
        (lmda == lmda_max) ?
        edpp_v1_0 :
        (edpp_resid_0 - resid) / lmda
    );
    vec_value_t v2 = edpp_resid_0 / lmda_next - resid / lmda;
    if (intercept) {
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
    ValueType lmda_next
)
{
    using state_t = std::decay_t<StateType>;
    using vec_value_t = typename state_t::vec_value_t;

    const auto& abs_grad = state.abs_grad;
    const auto lmda = state.lmda;
    const auto alpha = state.alpha;
    const auto& penalty = state.penalty;
    const auto& strong_hashset = state.strong_hashset;
    const auto& edpp_safe_set = state.edpp_safe_set;
    const auto delta_strong_size = state.delta_strong_size;
    const auto max_strong_size = state.max_strong_size;
    const auto strong_rule = state.strong_rule;
    auto& strong_set = state.strong_set;

    const auto old_strong_set_size = strong_set.size();
    const auto new_safe_size = edpp_safe_set.size() - old_strong_set_size;

    assert(strong_set.size() <= abs_grad.size());

    const auto is_strong = [&](auto i) { 
        return strong_hashset.find(i) != strong_hashset.end(); 
    };

    const auto do_fixed_greedy = [&]() {
        size_t size_capped = std::min(delta_strong_size, new_safe_size);

        strong_set.insert(strong_set.end(), size_capped, -1);
        const auto factor = (alpha <= 0) ? 1e-3 : alpha;
        const auto abs_grad_p = vec_value_t::NullaryExpr(
            abs_grad.size(), [&](auto i) {
                return (penalty[i] <= 0) ? 0.0 : abs_grad[i] / penalty[i];
            }
        ) / factor;
        // TODO: PARALLELIZE!
        util::k_imax(
            abs_grad_p, 
            is_strong,
            size_capped, 
            std::next(strong_set.begin(), old_strong_set_size)
        );
    };

    /* update strong_set */
    // Use either the fixed-increment rule or strong rule to increase strong set.
    if (strong_rule == state::strong_rule_type::_default) {
        const auto strong_rule_lmda = (2 * lmda_next - lmda) * alpha;

        // TODO: PARALLELIZE!
        for (int i = 0; i < abs_grad.size(); ++i) {
            if (is_strong(i)) continue;
            if (abs_grad[i] > strong_rule_lmda * penalty[i]) {
                strong_set.push_back(i);
            }
        }

        // If no new strong variables were added, need a fall-back.
        // Use fixed-greedy method.
        if (strong_set.size() == old_strong_set_size) {
            do_fixed_greedy();
        }
    } else if (strong_rule == state::strong_rule_type::_fixed_greedy) {
        do_fixed_greedy();
    } else if (strong_rule == state::strong_rule_type::_safe) {
        for (int i = 0; i < edpp_safe_set.size(); ++i) {
            if (is_strong(edpp_safe_set[i])) continue;
            strong_set.push_back(edpp_safe_set[i]);
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
    ValueType lmda_next
)
{
    screen_edpp(state, lmda_next);
    screen_strong(state, lmda_next);
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
template <class StateType, class StatePinNaiveType, class GradsType>
ADELIE_CORE_STRONG_INLINE
size_t kkt(
    StateType& state,
    const StatePinNaiveType& state_pin_naive,
    GradsType& grads
)
{
    const auto& X_means = state.X_means;
    const auto& groups = state.groups;
    const auto& group_sizes = state.group_sizes;
    const auto alpha = state.alpha;
    const auto& penalty = state.penalty;
    const auto intercept = state.intercept;
    const auto& strong_hashset = state.strong_hashset;
    auto& X = *state.X;

    const auto is_strong = [&](auto i) {
        return strong_hashset.find(i) != strong_hashset.end();
    };

    const auto n_lmdas = state_pin_naive.lmdas.size();

    // Keep track of the number of lmdas that are viable.
    // Viable means KKT hasn't failed yet for this lmda.
    int n_valid_solutions = n_lmdas;

    // Important to loop over groups first.
    // X usually loads new matrices when column blocks change.
    // If we loop over lmdas first, we will re-read column blocks multiple time.
    // Better to read once and process on all lmdas.

    // First, loop over non-strong set, compute gradients, and update n_valid_solutions.
    for (int k = 0; k < groups.size(); ++k) {
        if (is_strong(k)) continue;

        const auto gk = groups[k];
        const auto gk_size = group_sizes[k];
        const auto pk = penalty[k];

        for (int l = 0; l < n_valid_solutions; ++l) {
            auto grad_lk = grads[l].segment(gk, gk_size);

            X.bmul(gk, gk_size, state_pin_naive.resids[l], grad_lk);
            if (intercept) {
                grad_lk -= state_pin_naive.resid_sums[l] * X_means.segment(gk, gk_size);
            }
            const auto abs_grad_lk = grad_lk.matrix().norm();
            const auto lmda_l = state_pin_naive.lmdas[l];
            if (abs_grad_lk > lmda_l * alpha * pk * (1 + 1e-6)) {
                n_valid_solutions = l;
                break;
            }
        }
         
        if (n_valid_solutions <= 0) break;
    }

    if (n_valid_solutions <= 0) return 0;

    // If n_valid_solutions > 0, 
    // compute gradient for strong variables only for the last solution.
    for (int k = 0; k < groups.size(); ++k) {
        if (!is_strong(k)) continue;
        const auto l = n_valid_solutions-1;
        const auto gk = groups[k];
        const auto gk_size = group_sizes[k];
        auto grad_lk = grads[l].segment(gk, gk_size);
        X.bmul(gk, gk_size, state_pin_naive.resids[l], grad_lk);
        if (intercept) {
            grad_lk -= state_pin_naive.resid_sums[l] * X_means.segment(gk, gk_size);
        }
    }

    return n_valid_solutions;
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
    const auto delta_lmda_path_size = state.delta_lmda_path_size;
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
    auto& benchmark_fit = state.benchmark_fit;
    auto& benchmark_kkt = state.benchmark_kkt;
    auto& benchmark_invariance = state.benchmark_invariance;

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

    // if there is some large lambda or EDPP setup is needed, do initial fit.
    if (large_lmda_path_size || setup_edpp) {
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
    std::vector<vec_value_t> grads;
    sw_t sw;

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
            lmda_path_idx, 
            std::min(delta_lmda_path_size, lmda_path.size() - lmda_path_idx)
        );

        // ==================================================================================== 
        // Screening step
        // ==================================================================================== 
        sw.start();
        naive::screen(
            state,
            lmda_batch[0]
        );
        benchmark_screen.push_back(sw.elapsed());

        try {
            // ==================================================================================== 
            // Fit step
            // ==================================================================================== 
            // Save all current valid quantities that will be modified in-place by fit.
            // This is needed for the invariance step in case no valid solutions are found.
            save_prev_valid();
            sw.start();
            auto&& state_pin_naive = naive::fit(
                state,
                lmda_batch,
                update_coefficients_f,
                check_user_interrupt
            );
            benchmark_fit.push_back(sw.elapsed());

            // ==================================================================================== 
            // KKT step
            // ==================================================================================== 
            grads.resize(lmda_batch.size());
            for (auto& g : grads) g.resize(p);
            sw.start();
            const auto n_valid_solutions = naive::kkt(
                state,
                state_pin_naive,
                grads
            );
            benchmark_kkt.push_back(sw.elapsed());

            // ==================================================================================== 
            // Invariance step
            // ==================================================================================== 
            sw.start();
            lmda_path_idx += n_valid_solutions;
            // If no valid solutions found, restore to the previous valid state,
            // so that we can start over after screening more variables.
            if (n_valid_solutions <= 0) {
                load_prev_valid();
            // Otherwise, use the last valid state found in kkt.
            } else {
                const auto idx = n_valid_solutions - 1;
                resid.swap(state_pin_naive.resids[idx]);
                resid_sum = state_pin_naive.resid_sums[idx];
                // TODO: implement swap
                Eigen::Map<vec_value_t>(
                    strong_beta.data(),
                    strong_beta.size()
                ) = state_pin_naive.strong_betas[idx];
                Eigen::Map<vec_safe_bool_t>(
                    strong_is_active.data(), 
                    strong_is_active.size()
                ) = state_pin_naive.strong_is_actives[idx];
                rsq = state_pin_naive.rsqs[idx];
                lmda = lmda_batch[idx];
                grad.swap(grads[idx]);
                state::update_abs_grad(state);
            }
            naive::update_solutions(
                state, 
                state_pin_naive, 
                n_valid_solutions
            );
            benchmark_invariance.push_back(sw.elapsed());
        } catch (const std::exception& e) {
            load_prev_valid();
            throw util::propagator_error(e.what());
        }
    }
}

} // namespace naive
} // namespace solver
} // namespace adelie_core