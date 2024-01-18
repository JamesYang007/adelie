#pragma once
#include <adelie_core/optimization/search_pivot.hpp>
#include <adelie_core/solver/solver_gaussian_pin_naive.hpp>
#include <adelie_core/solver/utils.hpp>
#include <adelie_core/state/state_gaussian_naive.hpp>
#include <adelie_core/state/state_gaussian_pin_naive.hpp>
#include <adelie_core/util/algorithm.hpp>
#include <adelie_core/util/stopwatch.hpp>
#include <adelie_core/util/tqdm.hpp>

namespace adelie_core {
namespace solver {
namespace gaussian {
namespace naive {

template <class ValueType>
struct GaussianNaiveBufferPack
{
    using value_t = ValueType;
    using safe_bool_t = int8_t;
    using vec_value_t = util::rowvec_type<value_t>;
    using dyn_vec_value_t = std::vector<value_t>;
    using dyn_vec_bool_t = std::vector<safe_bool_t>;

    GaussianNaiveBufferPack(
        size_t n
    ):
        resid_prev(n)
    {}

    vec_value_t resid_prev;
    dyn_vec_value_t screen_beta_prev; 
    dyn_vec_bool_t screen_is_active_prev;
};

template <class StateType, class StateGaussianPinType>
ADELIE_CORE_STRONG_INLINE
void update_solutions(
    StateType& state,
    StateGaussianPinType& state_gaussian_pin_naive
)
{
    const auto y_var = state.y_var;
    auto& betas = state.betas;
    auto& rsqs = state.devs;
    auto& lmdas = state.lmdas;
    auto& intercepts = state.intercepts;

    betas.emplace_back(std::move(state_gaussian_pin_naive.betas.back()));
    intercepts.emplace_back(state_gaussian_pin_naive.intercepts.back());
    rsqs.emplace_back(state_gaussian_pin_naive.rsqs.back());
    lmdas.emplace_back(state_gaussian_pin_naive.lmdas.back());

    // normalize R^2 by null model
    rsqs.back() /= y_var;
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

template <class StateType,
          class BufferPackType,
          class ValueType,
          class UpdateCoefficientsType,
          class CUIType=util::no_op>
ADELIE_CORE_STRONG_INLINE
auto fit(
    StateType& state,
    BufferPackType& buffer_pack,
    ValueType lmda,
    UpdateCoefficientsType update_coefficients_f,
    CUIType check_user_interrupt = CUIType()
)
{
    using state_t = std::decay_t<StateType>;
    using value_t = typename state_t::value_t;
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
    const auto max_active_size = state.max_active_size;
    const auto max_iters = state.max_iters;
    const auto tol = state.tol;
    const auto adev_tol = state.adev_tol;
    const auto ddev_tol = state.ddev_tol;
    const auto newton_tol = state.newton_tol;
    const auto newton_max_iters = state.newton_max_iters;
    const auto n_threads = state.n_threads;
    auto& rsq = state.rsq;
    auto& resid_sum = state.resid_sum;
    auto& resid = state.resid;
    auto& screen_beta = state.screen_beta;
    auto& screen_is_active = state.screen_is_active;

    auto& resid_prev = buffer_pack.resid_prev;
    auto& screen_beta_prev = buffer_pack.screen_beta_prev;
    auto& screen_is_active_prev = buffer_pack.screen_is_active_prev;

    util::rowvec_type<value_t, 1> lmda_path;
    lmda_path = lmda;

    const auto save_prev_valid = [&]() {
        resid_prev = resid;
        screen_beta_prev = screen_beta;
        screen_is_active_prev = screen_is_active;
    };
    const auto load_prev_valid = [&]() {
        resid.swap(resid_prev);
        screen_beta.swap(screen_beta_prev);
        screen_is_active.swap(screen_is_active_prev);
    };

    save_prev_valid();

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
        intercept, max_active_size, max_iters, tol, adev_tol, ddev_tol, 
        newton_tol, newton_max_iters, n_threads,
        rsq,
        Eigen::Map<vec_value_t>(resid.data(), resid.size()),
        resid_sum,
        Eigen::Map<vec_value_t>(screen_beta.data(), screen_beta.size()), 
        Eigen::Map<vec_safe_bool_t>(screen_is_active.data(), screen_is_active.size())
    );

    try {
        pin::naive::solve(
            state_gaussian_pin_naive, 
            update_coefficients_f, 
            check_user_interrupt
        );
    } catch(...) {
        load_prev_valid();
        throw;
    }

    resid_sum = state_gaussian_pin_naive.resid_sum;
    rsq = state_gaussian_pin_naive.rsq;

    const auto screen_time = Eigen::Map<const util::rowvec_type<double>>(
        state_gaussian_pin_naive.benchmark_screen.data(),
        state_gaussian_pin_naive.benchmark_screen.size()
    ).sum();
    const auto active_time = Eigen::Map<const util::rowvec_type<double>>(
        state_gaussian_pin_naive.benchmark_active.data(),
        state_gaussian_pin_naive.benchmark_active.size()
    ).sum();

    return std::make_tuple(
        std::move(state_gaussian_pin_naive), 
        screen_time, 
        active_time
    );
}

/**
 * Checks the KKT condition on the proposed solution.
 */
template <class StateType, class ValueType>
ADELIE_CORE_STRONG_INLINE
size_t kkt(
    StateType& state,
    ValueType lmda
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
    const auto& resid = state.resid;
    const auto resid_sum = state.resid_sum;
    auto& X = *state.X;
    auto& grad = state.grad;

    const auto is_screen = [&](auto i) {
        return screen_hashset.find(i) != screen_hashset.end();
    };

    // NOTE: no matter what, every gradient must be computed for pivot method.
    X.mul(resid, grad);
    if (intercept) {
        matrix::dvsubi(grad, resid_sum * X_means, n_threads);
    }
    state::gaussian::update_abs_grad(state, lmda);

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
    const auto& rsqs = state.devs;
    const auto early_exit = state.early_exit;
    const auto max_screen_size = state.max_screen_size;
    const auto adev_tol = state.adev_tol;
    const auto ddev_tol = state.ddev_tol;
    const auto setup_lmda_max = state.setup_lmda_max;
    const auto setup_lmda_path = state.setup_lmda_path;
    const auto lmda_path_size = state.lmda_path_size;
    const auto min_ratio = state.min_ratio;
    const auto intercept = state.intercept;
    const auto n_threads = state.n_threads;
    const auto& screen_is_active = state.screen_is_active;
    const auto& abs_grad = state.abs_grad;
    const auto& resid = state.resid;
    const auto& resid_sum = state.resid_sum; // MUST be a reference since the most updated value is needed
    auto& X = *state.X;
    auto& lmda_max = state.lmda_max;
    auto& lmda_path = state.lmda_path;
    auto& grad = state.grad;
    auto& lmda = state.lmda;
    auto& benchmark_screen = state.benchmark_screen;
    auto& benchmark_fit_screen = state.benchmark_fit_screen;
    auto& benchmark_fit_active = state.benchmark_fit_active;
    auto& benchmark_kkt = state.benchmark_kkt;
    auto& benchmark_invariance = state.benchmark_invariance;
    auto& n_valid_solutions = state.n_valid_solutions;
    auto& active_sizes = state.active_sizes;
    auto& screen_sizes = state.screen_sizes;

    const auto n = X.rows();
    GaussianNaiveBufferPack<value_t> buffer_pack(n);

    if (screen_set.size() > max_screen_size) throw util::max_basil_screen_set();

    // ==================================================================================== 
    // Initial fit for lambda ~ infinity to setup lmda_max
    // ==================================================================================== 
    // Only unpenalized (l1) groups are active in this case.
    // State must include all unpenalized groups.
    // We solve for large lambda, then back-track the KKT condition to find the lambda
    // that leads to that solution where all penalized variables have 0 coefficient.
    if (setup_lmda_max) {
        const auto large_lmda = std::numeric_limits<value_t>::max();

        fit(
            state,
            buffer_pack,
            large_lmda,
            update_coefficients_f,
            check_user_interrupt
        );

        /* Invariance */
        lmda = large_lmda;
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
                return (penalty[i] <= 0.0) ? 0.0 : (abs_grad[i] / penalty[i]);
            }
        ).maxCoeff() / factor;
    }

    // ==================================================================================== 
    // Generate lambda path if needed
    // ==================================================================================== 
    if (setup_lmda_path) {
        if (lmda_path_size <= 0) throw std::runtime_error("lmda_path_size must be > 0.");

        lmda_path.resize(lmda_path_size);

        generate_lmda_path(lmda_path, min_ratio, lmda_max);
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
            auto tup = fit(
                state, 
                buffer_pack,
                large_lmda_path[i], 
                update_coefficients_f, 
                check_user_interrupt
            );
            auto&& state_gaussian_pin_naive = std::get<0>(tup);

            /* Invariance */
            // save only the solutions that the user asked for (up to and not including lmda_max)
            if (i < large_lmda_path.size()-1) {
                update_solutions(
                    state, 
                    state_gaussian_pin_naive
                );
            // otherwise, put the state at the last fitted lambda (lmda_max)
            } else {
                lmda = large_lmda_path[i];
                X.mul(resid, grad);
                if (intercept) {
                    matrix::dvsubi(grad, resid_sum * X_means, n_threads);
                }
                state::gaussian::update_abs_grad(state, lmda);
            }
        }
    }

    size_t lmda_path_idx = rsqs.size(); // next index into lmda_path to fit

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
    const auto pb_add_suffix = [&]() {
        // print extra information with the progress bar
        if (display) {
            // current training R^2
            pb << " [dev:" 
                << std::fixed << std::setprecision(1) 
                << ((rsqs.size() == 0) ? 0.0 : rsqs.back()) * 100
                << "%]"
                ; 
        }
    };

    for (int _ : pb)
    {
        static_cast<void>(_);

        // check early exit
        if (early_exit && (rsqs.size() >= 2)) {
            const auto rsq_u = rsqs[rsqs.size()-1];
            const auto rsq_m = rsqs[rsqs.size()-2];
            if ((rsq_u >= adev_tol) || (rsq_u-rsq_m <= ddev_tol)) 
            {
                pb_add_suffix();
                break;
            }
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
                screen(
                    state,
                    lmda_curr,
                    kkt_passed,
                    n_new_active
                );
                state::gaussian::naive::update_screen_derived(state);
                benchmark_screen.push_back(sw.elapsed());

                // ==================================================================================== 
                // Fit step
                // ==================================================================================== 
                // Save all current valid quantities that will be modified in-place by fit.
                // This is needed in case we exit with exception and need to restore invariance.
                auto tup = fit(
                    state,
                    buffer_pack,
                    lmda_curr,
                    update_coefficients_f,
                    check_user_interrupt
                );
                auto&& state_gaussian_pin_naive = std::get<0>(tup);
                benchmark_fit_screen.push_back(std::get<1>(tup));
                benchmark_fit_active.push_back(std::get<2>(tup));

                // ==================================================================================== 
                // KKT step
                // ==================================================================================== 
                sw.start();
                kkt_passed = kkt(
                    state,
                    lmda_curr
                );
                benchmark_kkt.push_back(sw.elapsed());
                n_valid_solutions.push_back(kkt_passed);

                // ==================================================================================== 
                // Invariance step
                // ==================================================================================== 
                sw.start();
                lmda_path_idx += kkt_passed;
                lmda = lmda_curr;
                if (kkt_passed) {
                    update_solutions(
                        state, 
                        state_gaussian_pin_naive
                    );
                }
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
            } catch (...) {
                pb_add_suffix();
                throw;
            }

            if (kkt_passed) break;
        } // end while(1)

        pb_add_suffix();
    }
}

} // namespace naive
} // namespace gaussian
} // namespace solver
} // namespace adelie_core