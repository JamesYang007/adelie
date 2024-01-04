#pragma once
#include <adelie_core/optimization/search_pivot.hpp>
#include <adelie_core/solver/solve_pin_naive.hpp>
#include <adelie_core/state/state_pin_naive.hpp>
#include <adelie_core/util/algorithm.hpp>
#include <adelie_core/util/stopwatch.hpp>

namespace adelie_core {
namespace solver {
namespace naive {

template <class StateType,
          class UpdateCoefficientsType,
          class CUIType=util::no_op>
inline void solve_glm(
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


}

} // namespace naive 
} // namespace solver
} // namespace adelie_core