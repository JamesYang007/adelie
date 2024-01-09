#pragma once
#include <adelie_core/optimization/search_pivot.hpp>
#include <adelie_core/solver/solver_gaussian_pin_naive.hpp>
#include <adelie_core/state/state_gaussian_naive.hpp>
#include <adelie_core/state/state_gaussian_pin_naive.hpp>
#include <adelie_core/util/algorithm.hpp>
#include <adelie_core/util/stopwatch.hpp>
#include <adelie_core/util/tqdm.hpp>

namespace adelie_core {
namespace solver {
namespace gaussian {
namespace naive {

template <class ValueType, class IndexType> 
inline
auto objective(
    ValueType beta0, 
    const Eigen::Ref<const util::rowvec_type<ValueType>>& beta,
    const Eigen::Ref<const util::rowmat_type<ValueType>>& X,
    const Eigen::Ref<const util::rowvec_type<ValueType>>& y,
    const Eigen::Ref<const util::rowvec_type<IndexType>>& groups,
    const Eigen::Ref<const util::rowvec_type<IndexType>>& group_sizes,
    ValueType lmda,
    ValueType alpha,
    const Eigen::Ref<const util::rowvec_type<ValueType>>& penalty,
    const Eigen::Ref<const util::rowvec_type<ValueType>>& weights
)
{
    ValueType p_ = 0.0;
    for (int j = 0; j < groups.size(); ++j) {
        const auto begin = groups[j];
        const auto size = group_sizes[j];
        const auto b_norm2 = beta.segment(begin, size).matrix().norm();
        p_ += penalty[j] * b_norm2 * (
            alpha + 0.5 * (1-alpha) * b_norm2
        );
    }
    p_ *= lmda;
    util::rowvec_type<ValueType> resid = (y.matrix() - beta.matrix() * X.transpose()).array() - beta0;
    return 0.5 * (weights * resid.square()).sum() + p_;
}

template <class StateType, class StateGaussianPinType>
ADELIE_CORE_STRONG_INLINE
void update_solutions(
    StateType& state,
    StateGaussianPinType& state_gaussian_pin_naive,
    size_t n_sols
)
{
    const auto y_var = state.y_var;
    auto& betas = state.betas;
    auto& rsqs = state.rsqs;
    auto& lmdas = state.lmdas;
    auto& intercepts = state.intercepts;

    for (size_t i = 0; i < n_sols; ++i) {
        betas.emplace_back(std::move(state_gaussian_pin_naive.betas[i]));
        intercepts.emplace_back(state_gaussian_pin_naive.intercepts[i]);
        rsqs.emplace_back(state_gaussian_pin_naive.rsqs[i]);
        lmdas.emplace_back(state_gaussian_pin_naive.lmdas[i]);

        // normalize R^2 by null model
        rsqs.back() /= y_var;
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

        // add everything beyond the cutoff index that isn't strong yet
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
        if ((screen_set.size() == old_screen_set_size) && !all_kkt_passed) {
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
        throw std::runtime_error("Unknown strong rule!");
    }

    // If adding new amount went over max strong size, 
    // undo the change to keep invariance from before, then throw exception.
    if (screen_set.size() > max_screen_size) {
        screen_set.erase(
            std::next(screen_set.begin(), old_screen_set_size),
            screen_set.end()
        );
        throw util::max_basil_screen_set();
    }

    /* update derived strong quantities */
    state::gaussian::naive::update_screen_derived(state);

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
    using matrix_naive_t = typename state_t::matrix_t;
    using state_gaussian_pin_naive_t = state::StateGaussianPinNaive<
        matrix_naive_t,
        typename std::decay_t<matrix_naive_t>::value_t,
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
    const auto& weights = state.weights;
    const auto& screen_set = state.screen_set;
    const auto& screen_g1 = state.screen_g1;
    const auto& screen_g2 = state.screen_g2;
    const auto& screen_begins = state.screen_begins;
    const auto& screen_vars = state.screen_vars;
    const auto& screen_X_means = state.screen_X_means;
    const auto& screen_transforms = state.screen_transforms;
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
    auto& screen_beta = state.screen_beta;
    auto& screen_is_active = state.screen_is_active;

    state_gaussian_pin_naive_t state_gaussian_pin_naive(
        X,
        y_mean,
        y_var,
        groups, 
        group_sizes,
        alpha, 
        penalty,
        weights,
        Eigen::Map<const vec_index_t>(screen_set.data(), screen_set.size()), 
        Eigen::Map<const vec_index_t>(screen_g1.data(), screen_g1.size()), 
        Eigen::Map<const vec_index_t>(screen_g2.data(), screen_g2.size()), 
        Eigen::Map<const vec_index_t>(screen_begins.data(), screen_begins.size()), 
        Eigen::Map<const vec_value_t>(screen_vars.data(), screen_vars.size()), 
        Eigen::Map<const vec_value_t>(screen_X_means.data(), screen_X_means.size()), 
        screen_transforms,
        lmda_path,
        intercept, max_iters, tol, rsq_tol, rsq_slope_tol, rsq_curv_tol, 
        newton_tol, newton_max_iters, n_threads,
        rsq,
        Eigen::Map<vec_value_t>(resid.data(), resid.size()),
        resid_sum,
        Eigen::Map<vec_value_t>(screen_beta.data(), screen_beta.size()), 
        Eigen::Map<vec_safe_bool_t>(screen_is_active.data(), screen_is_active.size())
    );

    pin::naive::solve(
        state_gaussian_pin_naive, 
        update_coefficients_f, 
        check_user_interrupt
    );

    return state_gaussian_pin_naive;
}

/**
 * Checks the KKT condition on the proposed solutions in state_gaussian_pin_naive.
 */
template <class StateType, class StateGaussianPinNaiveType>
ADELIE_CORE_STRONG_INLINE
size_t kkt(
    StateType& state,
    const StateGaussianPinNaiveType& state_gaussian_pin_naive
)
{
    const auto& X_means = state.X_means;
    const auto& groups = state.groups;
    const auto alpha = state.alpha;
    const auto& penalty = state.penalty;
    const auto intercept = state.intercept;
    const auto n_threads = state.n_threads;
    const auto& screen_hashset = state.screen_hashset;
    const auto& abs_grad = state.abs_grad;
    const auto lmda = state_gaussian_pin_naive.lmdas[0];
    auto& X = *state.X;
    auto& grad = state.grad;

    const auto is_screen = [&](auto i) {
        return screen_hashset.find(i) != screen_hashset.end();
    };

    // NOTE: no matter what, every gradient must be computed for pivot method.
    X.mul(state_gaussian_pin_naive.resids[0], grad);
    if (intercept) {
        matrix::dvsubi(grad, state_gaussian_pin_naive.resid_sums[0] * X_means, n_threads);
    }
    state::gaussian::update_abs_grad(state, lmda);

    // First, loop over non-strong set, compute gradients, and update n_valid_solutions.
    for (int k = 0; k < groups.size(); ++k) {
        if (is_screen(k)) continue;
        const auto pk = penalty[k];
        const auto abs_grad_k = abs_grad[k];
        if (abs_grad_k > lmda * alpha * pk) return false;
    }

    return true;
}

template <class StateType,
          class UpdateCoefficientsType,
          class CUIType=util::no_op>
inline void solve(
    StateType&& state,
    bool display,
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
    const auto& screen_set = state.screen_set;
    const auto& rsqs = state.rsqs;
    const auto early_exit = state.early_exit;
    const auto max_screen_size = state.max_screen_size;
    const auto rsq_tol = state.rsq_tol;
    const auto rsq_slope_tol = state.rsq_slope_tol;
    const auto rsq_curv_tol = state.rsq_curv_tol;
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
    auto& screen_is_active = state.screen_is_active;
    auto& screen_beta = state.screen_beta;
    auto& resid = state.resid;
    auto& resid_sum = state.resid_sum;
    auto& grad = state.grad;
    auto& rsq = state.rsq;
    auto& lmda = state.lmda;
    auto& resid_prev_valid = state.resid_prev_valid;
    auto& screen_beta_prev_valid = state.screen_beta_prev_valid; 
    auto& screen_is_active_prev_valid = state.screen_is_active_prev_valid;
    auto& benchmark_screen = state.benchmark_screen;
    auto& benchmark_fit_screen = state.benchmark_fit_screen;
    auto& benchmark_fit_active = state.benchmark_fit_active;
    auto& benchmark_kkt = state.benchmark_kkt;
    auto& benchmark_invariance = state.benchmark_invariance;
    auto& n_valid_solutions = state.n_valid_solutions;
    auto& active_sizes = state.active_sizes;
    auto& screen_sizes = state.screen_sizes;

    if (screen_set.size() > max_screen_size) throw util::max_basil_screen_set();

    const auto save_prev_valid = [&]() {
        resid_prev_valid = resid;
        screen_beta_prev_valid = screen_beta;
        screen_is_active_prev_valid = screen_is_active;
    };
    const auto load_prev_valid = [&]() {
        resid.swap(resid_prev_valid);
        screen_beta.swap(screen_beta_prev_valid);
        screen_is_active.swap(screen_is_active_prev_valid);
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
            auto&& state_gaussian_pin_naive = fit(
                state,
                large_lmda_path,
                update_coefficients_f,
                check_user_interrupt
            );

            /* Invariance */
            resid_sum = state_gaussian_pin_naive.resid_sums.back();
            rsq = state_gaussian_pin_naive.rsqs.back();
            lmda = state_gaussian_pin_naive.lmdas.back();
            X.mul(resid, grad);
            if (intercept) {
                matrix::dvsubi(grad, resid_sum * X_means, n_threads);
            }
            state::gaussian::update_abs_grad(state, lmda);

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
    // If no lambda in lmda_path is > lmda_max, state is left unchanged.
    // Otherwise, it is in its invariance at lmda = lmda_max.
    // All solutions to lambda > lambda_max are saved.

    // slice lambda_path up to lmda_max
    const auto large_lmda_path_size = std::find_if(
        lmda_path.data(), 
        lmda_path.data() + lmda_path.size(),
        [&](auto x) { return x <= lmda_max; }
    ) - lmda_path.data();

    // if there is some large lambda or lmda_max setup is needed
    // do the initial fit up to (and including) lmda_max.
    if (large_lmda_path_size || setup_lmda_max) {
        // create a lambda path containing only lmdas > lambda_max
        // and additionally lambda_max at the end.
        // If large_lmda_path_size > 0, mind as well fit for lambda_max as well to go down the path.
        vec_value_t large_lmda_path(large_lmda_path_size + 1);
        large_lmda_path.head(large_lmda_path_size) = lmda_path.head(large_lmda_path_size);
        large_lmda_path[large_lmda_path_size] = lmda_max;

        try {
            save_prev_valid();
            auto&& state_gaussian_pin_naive = fit(
                state, 
                large_lmda_path, 
                update_coefficients_f, 
                check_user_interrupt
            );

            /* Invariance */
            // put the state at the last fitted lambda (should be lmda_max)
            // but save only the solutions that the user asked for (up to and not including lmda_max).
            resid.swap(state_gaussian_pin_naive.resids.back());
            resid_sum = state_gaussian_pin_naive.resid_sums.back();
            // TODO: implement swap
            Eigen::Map<vec_value_t>(
                screen_beta.data(),
                screen_beta.size()
            ) = state_gaussian_pin_naive.screen_betas.back();
            Eigen::Map<vec_safe_bool_t>(
                screen_is_active.data(), 
                screen_is_active.size()
            ) = state_gaussian_pin_naive.screen_is_actives.back();
            rsq = state_gaussian_pin_naive.rsqs.back();
            lmda = state_gaussian_pin_naive.lmdas.back();
            X.mul(resid, grad);
            if (intercept) {
                matrix::dvsubi(grad, resid_sum * X_means, n_threads);
            }
            state::gaussian::update_abs_grad(state, lmda);
            update_solutions(
                state, 
                state_gaussian_pin_naive, 
                state_gaussian_pin_naive.lmdas.size()-1
            );
        } catch (const std::exception& e) {
            load_prev_valid();
            throw util::propagator_error(e.what());
        }
    }

    size_t lmda_path_idx = rsqs.size(); // next index into lmda_path to fit

    // ==================================================================================== 
    // BASIL iterations for lambda <= lambda_max
    // ==================================================================================== 
    // In this case, screen_set may not contain the true active set.
    // We must go through BASIL iterations to solve each lambda.
    vec_value_t lmda_batch;
    sw_t sw;
    int current_active_size = Eigen::Map<vec_safe_bool_t>(
        screen_is_active.data(),
        screen_is_active.size()
    ).sum();
    bool kkt_passed = true;
    int n_new_active = 0;

    auto pb = util::tq::trange(lmda_path.size() - lmda_path_idx);
    pb.set_display(display);

    for (int _ : pb)
    {
        // print extra information with the progress bar
        if (display) {
            // current training R^2
            pb << " [dev:" 
                << std::fixed << std::setprecision(1) 
                << ((rsqs.size() == 0) ? 0.0 : rsqs.back()) * 100
                << "%]"
                ; 
        }

        // check early exit
        if (early_exit && (rsqs.size() >= 3)) {
            const auto rsq_u = rsqs[rsqs.size()-1];
            const auto rsq_m = rsqs[rsqs.size()-2];
            const auto rsq_l = rsqs[rsqs.size()-3];
            if (pin::check_early_stop_rsq(rsq_l, rsq_m, rsq_u, rsq_slope_tol, rsq_curv_tol) || 
                (rsq_u >= rsq_tol)) break;
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
        screen(
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
            auto&& state_gaussian_pin_naive = fit(
                state,
                lmda_batch,
                update_coefficients_f,
                check_user_interrupt
            );
            benchmark_fit_screen.push_back(
                Eigen::Map<const util::rowvec_type<double>>(
                    state_gaussian_pin_naive.benchmark_screen.data(),
                    state_gaussian_pin_naive.benchmark_screen.size()
                ).sum()
            );
            benchmark_fit_active.push_back(
                Eigen::Map<const util::rowvec_type<double>>(
                    state_gaussian_pin_naive.benchmark_active.data(),
                    state_gaussian_pin_naive.benchmark_active.size()
                ).sum()
            );

            // ==================================================================================== 
            // KKT step
            // ==================================================================================== 
            sw.start();
            kkt_passed = kkt(
                state,
                state_gaussian_pin_naive
            );
            benchmark_kkt.push_back(sw.elapsed());
            n_valid_solutions.push_back(kkt_passed);

            // ==================================================================================== 
            // Invariance step
            // ==================================================================================== 
            sw.start();
            lmda_path_idx += kkt_passed;
            resid_sum = state_gaussian_pin_naive.resid_sums[0];
            rsq = state_gaussian_pin_naive.rsqs[0];
            lmda = lmda_batch[0];
            update_solutions(
                state, 
                state_gaussian_pin_naive, 
                kkt_passed
            );
            benchmark_invariance.push_back(sw.elapsed());

            // ==================================================================================== 
            // Diagnostic step
            // ==================================================================================== 
            if (kkt_passed) {
                active_sizes.push_back(state_gaussian_pin_naive.active_set.size());
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
        } catch (const std::exception& e) {
            load_prev_valid();
            throw util::propagator_error(e.what());
        }
    }
}

} // namespace naive
} // namespace gaussian
} // namespace solver
} // namespace adelie_core