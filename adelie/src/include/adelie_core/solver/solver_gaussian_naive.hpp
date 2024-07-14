#pragma once
#include <adelie_core/solver/solver_base.hpp>
#include <adelie_core/solver/solver_gaussian_pin_naive.hpp>
#include <adelie_core/state/state_gaussian_naive.hpp>
#include <adelie_core/state/state_gaussian_pin_naive.hpp>

namespace adelie_core {
namespace solver {
namespace gaussian {
namespace naive {

template <class ValueType, class SafeBoolType=int8_t>
struct GaussianNaiveBufferPack
{
    using value_t = ValueType;
    using safe_bool_t = SafeBoolType;
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

template <class StateType, class StateGaussianPinType, class ValueType>
ADELIE_CORE_STRONG_INLINE
void update_solutions(
    StateType& state,
    StateGaussianPinType& state_gaussian_pin_naive,
    ValueType lmda
)
{
    const auto y_var = state.y_var;
    auto& betas = state.betas;
    auto& duals = state.duals;
    auto& devs = state.devs;
    auto& lmdas = state.lmdas;
    auto& intercepts = state.intercepts;

    betas.emplace_back(std::move(state_gaussian_pin_naive.betas.back()));
    duals.emplace_back(std::move(state_gaussian_pin_naive.duals.back()));
    intercepts.emplace_back(state_gaussian_pin_naive.intercepts.back());
    lmdas.emplace_back(lmda);

    const auto dev = state_gaussian_pin_naive.rsqs.back();
    devs.emplace_back(dev / y_var);
}

template <class StateType,
          class BufferPackType,
          class ValueType,
          class CUIType=util::no_op>
ADELIE_CORE_STRONG_INLINE
auto fit(
    StateType& state,
    BufferPackType& buffer_pack,
    ValueType lmda,
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
    using constraint_t = typename state_t::constraint_t;
    using matrix_naive_t = typename state_t::matrix_t;
    using state_gaussian_pin_naive_t = state::StateGaussianPinNaive<
        constraint_t,
        matrix_naive_t,
        typename std::decay_t<matrix_naive_t>::value_t,
        index_t,
        safe_bool_t
    >;

    auto& X = *state.X;
    const auto y_mean = state.y_mean;
    const auto y_var = state.y_var;
    const auto& constraints = state.constraints;
    const auto& groups = state.groups;
    const auto& group_sizes = state.group_sizes;
    const auto& dual_groups = state.dual_groups;
    const auto alpha = state.alpha;
    const auto& penalty = state.penalty;
    const auto& weights = state.weights;
    const auto& screen_set = state.screen_set;
    const auto& screen_begins = state.screen_begins;
    const auto& screen_vars = state.screen_vars;
    const auto& screen_X_means = state.screen_X_means;
    const auto& screen_transforms = state.screen_transforms;
    const auto constraint_buffer_size = state.constraint_buffer_size;
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
    auto& active_set_size = state.active_set_size;
    auto& active_set = state.active_set;

    auto& resid_prev = buffer_pack.resid_prev;
    auto& screen_beta_prev = buffer_pack.screen_beta_prev;
    auto& screen_is_active_prev = buffer_pack.screen_is_active_prev;

    util::rowvec_type<value_t, 1> lmda_path;
    lmda_path = lmda;

    // Save all current valid quantities that will be modified in-place by fit.
    // This is needed in case we exit with exception and need to restore invariance.
    // Saving SHOULD NOT swap since we still need the values of the current containers.
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
        constraints,
        groups, 
        group_sizes,
        dual_groups,
        alpha, 
        penalty,
        weights,
        Eigen::Map<const vec_index_t>(screen_set.data(), screen_set.size()), 
        Eigen::Map<const vec_index_t>(screen_begins.data(), screen_begins.size()), 
        Eigen::Map<const vec_value_t>(screen_vars.data(), screen_vars.size()), 
        Eigen::Map<const vec_value_t>(screen_X_means.data(), screen_X_means.size()), 
        screen_transforms,
        lmda_path,
        constraint_buffer_size,
        intercept, max_active_size, max_iters, 
        // TODO: still unclear whether we should be max'ing or not.
        // tolerance is relative to the scaling of null deviance and current total weight sum (== 1)
        tol * std::max<value_t>(y_var, 1), 
        adev_tol, ddev_tol, 
        newton_tol, newton_max_iters, n_threads,
        rsq,
        Eigen::Map<vec_value_t>(resid.data(), resid.size()),
        resid_sum,
        Eigen::Map<vec_value_t>(screen_beta.data(), screen_beta.size()), 
        Eigen::Map<vec_safe_bool_t>(screen_is_active.data(), screen_is_active.size()),
        active_set_size,
        active_set
    );

    try {
        pin::naive::solve(
            state_gaussian_pin_naive, 
            check_user_interrupt
        );
    } catch(...) {
        load_prev_valid();
        throw;
    }

    resid_sum = state_gaussian_pin_naive.resid_sum;
    rsq = state_gaussian_pin_naive.rsq;
    active_set_size = state_gaussian_pin_naive.active_set_size;

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

template <class StateType,
          class PBType,
          class ExitCondType,
          class TidyType,
          class CUIType>
inline void solve(
    StateType&& state,
    PBType&& pb,
    ExitCondType exit_cond_f,
    TidyType tidy_f,
    CUIType check_user_interrupt
)
{
    using state_t = std::decay_t<StateType>;
    using value_t = typename state_t::value_t;
    using safe_bool_t = typename state_t::safe_bool_t;

    const auto n = state.X->rows();
    GaussianNaiveBufferPack<value_t, safe_bool_t> buffer_pack(n);

    const auto pb_add_suffix_f = [&](const auto& state, auto& pb) {
        solver::pb_add_suffix(state, pb);
    };
    const auto update_loss_null_f = [](const auto&) {};
    const auto update_invariance_f = [&](auto& state, const auto&, auto lmda) {
        const auto& X_means = state.X_means;
        const auto& weights = state.weights;
        const auto intercept = state.intercept;
        const auto n_threads = state.n_threads;
        const auto& resid = state.resid;
        const auto& resid_sum = state.resid_sum; // MUST be a reference since the most updated value is needed
        auto& X = *state.X;
        auto& grad = state.grad;

        state.lmda = lmda;
        X.mul(resid, weights, grad);
        if (intercept) {
            matrix::dvsubi(grad, resid_sum * X_means, n_threads);
        }
        state::update_abs_grad(state, lmda);
    };
    const auto update_solutions_f = [&](auto& state, auto& state_gaussian_pin_naive, auto lmda) {
        update_solutions(
            state, 
            state_gaussian_pin_naive,
            lmda
        );
        tidy_f();
    };
    const auto early_exit_f = [&](const auto& state) {
        return solver::early_exit(state) || exit_cond_f();
    };
    const auto screen_f = [](auto& state, auto lmda, auto kkt_passed, auto n_new_active) {
        solver::screen(
            state,
            lmda,
            kkt_passed,
            n_new_active
        );
        state::gaussian::naive::update_screen_derived(state);
    };
    const auto fit_f = [&](auto& state, auto lmda) {
        return fit(
            state, 
            buffer_pack, 
            lmda, 
            check_user_interrupt
        );
    };

    solver::solve_core(
        state,
        pb,
        pb_add_suffix_f,
        update_loss_null_f,
        update_invariance_f,
        update_solutions_f,
        early_exit_f,
        screen_f,
        fit_f
    );
}

template <class StateType,
          class PBType,
          class ExitCondType,
          class CUIType=util::no_op>
inline void solve(
    StateType&& state,
    PBType&& pb,
    ExitCondType exit_cond_f,
    CUIType check_user_interrupt = CUIType()
)
{
    solve(
        state,
        pb,
        exit_cond_f,
        [](){},
        check_user_interrupt
    );
}

} // namespace naive
} // namespace gaussian
} // namespace solver
} // namespace adelie_core