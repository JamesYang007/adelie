#pragma once
#include <adelie_core/solver/solver_base.hpp>
#include <adelie_core/solver/solver_gaussian_pin_cov.hpp>
#include <adelie_core/state/state_gaussian_cov.hpp>
#include <adelie_core/state/state_gaussian_pin_cov.hpp>
#include <adelie_core/matrix/utils.hpp>

namespace adelie_core {
namespace solver {
namespace gaussian {
namespace cov {

template <class ValueType>
struct GaussianCovBufferPack
{
    using value_t = ValueType;
    using safe_bool_t = int8_t;
    using vec_value_t = util::rowvec_type<value_t>;
    using dyn_vec_value_t = std::vector<value_t>;
    using dyn_vec_bool_t = std::vector<safe_bool_t>;

    GaussianCovBufferPack(
        size_t p
    ):
        buffer_p(p)
    {}

    dyn_vec_value_t screen_grad_prev;
    dyn_vec_value_t screen_beta_prev; 
    dyn_vec_bool_t screen_is_active_prev;

    vec_value_t buffer_p;
};

template <class StateType, class PBType>
ADELIE_CORE_STRONG_INLINE
void pb_add_suffix(
    const StateType& state,
    PBType& pb
)
{
    const auto& devs = state.devs;
    // current relative training % dev explained
    const auto rdev = (
        (devs.size() < 2) ? 0.0 :
        (devs[devs.size()-1] - devs[devs.size()-2]) / devs[devs.size()-1]
    );
    pb << " [rdev:" 
        << std::fixed << std::setprecision(1) 
        << rdev * 100
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

    if (!early_exit || devs.size() < 2) return false;

    const auto rdev_tol = state.rdev_tol;

    const auto dev_u = devs[devs.size()-1];
    const auto dev_m = devs[devs.size()-2];
    if (dev_u-dev_m <= rdev_tol * dev_u) return true;

    return false;
}

template <class StateType, class StateGaussianPinType, class ValueType>
ADELIE_CORE_STRONG_INLINE
void update_solutions(
    StateType& state,
    StateGaussianPinType& state_gaussian_pin_cov,
    ValueType lmda
)
{
    auto& betas = state.betas;
    auto& intercepts = state.intercepts;
    auto& devs = state.devs;
    auto& lmdas = state.lmdas;

    betas.emplace_back(std::move(state_gaussian_pin_cov.betas.back()));
    intercepts.emplace_back(0);
    lmdas.emplace_back(lmda);

    const auto dev = state_gaussian_pin_cov.rsqs.back();
    devs.emplace_back(dev);
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
    using matrix_cov_t = typename state_t::matrix_t;
    using state_gaussian_pin_cov_t = state::StateGaussianPinCov<
        matrix_cov_t,
        typename std::decay_t<matrix_cov_t>::value_t,
        index_t,
        safe_bool_t
    >;

    auto& A = *state.A;
    const auto& groups = state.groups;
    const auto& group_sizes = state.group_sizes;
    const auto alpha = state.alpha;
    const auto& penalty = state.penalty;
    const auto& screen_set = state.screen_set;
    const auto& screen_g1 = state.screen_g1;
    const auto& screen_g2 = state.screen_g2;
    const auto& screen_begins = state.screen_begins;
    const auto& screen_vars = state.screen_vars;
    const auto& screen_transforms = state.screen_transforms;
    const auto max_active_size = state.max_active_size;
    const auto max_iters = state.max_iters;
    const auto tol = state.tol;
    const auto rdev_tol = state.rdev_tol;
    const auto newton_tol = state.newton_tol;
    const auto newton_max_iters = state.newton_max_iters;
    const auto n_threads = state.n_threads;
    auto& rsq = state.rsq;
    auto& screen_beta = state.screen_beta;
    auto& screen_grad = state.screen_grad;
    auto& screen_is_active = state.screen_is_active;

    auto& screen_grad_prev = buffer_pack.screen_grad_prev;
    auto& screen_beta_prev = buffer_pack.screen_beta_prev;
    auto& screen_is_active_prev = buffer_pack.screen_is_active_prev;

    util::rowvec_type<value_t, 1> lmda_path;
    lmda_path = lmda;

    // Save all current valid quantities that will be modified in-place by fit.
    // This is needed in case we exit with exception and need to restore invariance.
    // Saving SHOULD NOT swap since we still need the values of the current containers.
    const auto save_prev_valid = [&]() {
        screen_grad_prev = screen_grad;
        screen_beta_prev = screen_beta;
        screen_is_active_prev = screen_is_active;
    };
    const auto load_prev_valid = [&]() {
        screen_grad.swap(screen_grad_prev);
        screen_beta.swap(screen_beta_prev);
        screen_is_active.swap(screen_is_active_prev);
    };

    save_prev_valid();

    state_gaussian_pin_cov_t state_gaussian_pin_cov(
        A,
        groups, 
        group_sizes,
        alpha, 
        penalty,
        Eigen::Map<const vec_index_t>(screen_set.data(), screen_set.size()), 
        Eigen::Map<const vec_index_t>(screen_g1.data(), screen_g1.size()), 
        Eigen::Map<const vec_index_t>(screen_g2.data(), screen_g2.size()), 
        Eigen::Map<const vec_index_t>(screen_begins.data(), screen_begins.size()), 
        Eigen::Map<const vec_value_t>(screen_vars.data(), screen_vars.size()), 
        screen_transforms,
        lmda_path,
        max_active_size, max_iters, tol, rdev_tol,
        newton_tol, newton_max_iters, n_threads,
        rsq,
        Eigen::Map<vec_value_t>(screen_beta.data(), screen_beta.size()), 
        Eigen::Map<vec_value_t>(screen_grad.data(), screen_grad.size()), 
        Eigen::Map<vec_safe_bool_t>(screen_is_active.data(), screen_is_active.size())
    );

    try {
        pin::cov::solve(
            state_gaussian_pin_cov, 
            update_coefficients_f, 
            check_user_interrupt
        );
    } catch(...) {
        load_prev_valid();
        throw;
    }

    rsq = state_gaussian_pin_cov.rsq;

    const auto screen_time = Eigen::Map<const util::rowvec_type<double>>(
        state_gaussian_pin_cov.benchmark_screen.data(),
        state_gaussian_pin_cov.benchmark_screen.size()
    ).sum();
    const auto active_time = Eigen::Map<const util::rowvec_type<double>>(
        state_gaussian_pin_cov.benchmark_active.data(),
        state_gaussian_pin_cov.benchmark_active.size()
    ).sum();

    return std::make_tuple(
        std::move(state_gaussian_pin_cov), 
        screen_time, 
        active_time
    );
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

    const auto p = state.A->cols();
    GaussianCovBufferPack<value_t> buffer_pack(p);

    const auto pb_add_suffix_f = [&](const auto& state, auto& pb) {
        if (display) cov::pb_add_suffix(state, pb);
    };
    const auto update_loss_null_f = [](const auto&) {};
    const auto update_invariance_f = [&](auto& state, auto lmda) {
        const auto& v = state.v;
        const auto& groups = state.groups;
        const auto& group_sizes = state.group_sizes;
        const auto& screen_set = state.screen_set;
        const auto& screen_beta = state.screen_beta;
        const auto& screen_begins = state.screen_begins;
        const auto& screen_is_active = state.screen_is_active;
        const auto n_threads = state.n_threads;
        auto& A = *state.A;
        auto& grad = state.grad;
        auto& buffer_p = buffer_pack.buffer_p;

        state.lmda = lmda;
        grad = v;
        for (int ss_idx = 0; ss_idx < screen_set.size(); ++ss_idx) {
            if (!screen_is_active[ss_idx]) continue;
            const auto ss = screen_set[ss_idx];
            const auto sb = screen_begins[ss_idx];
            const auto g = groups[ss];
            const auto gs = group_sizes[ss];
            const Eigen::Map<const vec_value_t> beta_g(
                screen_beta.data() + sb,
                gs
            );
            A.mul(g, gs, beta_g, buffer_p);
            matrix::dvsubi(grad, buffer_p, n_threads);
        }
        state::update_abs_grad(state, lmda);
    };
    const auto update_solutions_f = [](auto& state, auto& state_gaussian_pin_cov, auto lmda) {
        update_solutions(
            state, 
            state_gaussian_pin_cov,
            lmda
        );
    };
    const auto early_exit_f = [](const auto& state) {
        return cov::early_exit(state);
    };
    const auto screen_f = [](auto& state, auto lmda, auto kkt_passed, auto n_new_active) {
        solver::screen(
            state,
            lmda,
            kkt_passed,
            n_new_active
        );
        state::gaussian::cov::update_screen_derived(state);
    };
    const auto fit_f = [&](auto& state, auto lmda) {
        return fit(
            state, 
            buffer_pack, 
            lmda, 
            update_coefficients_f, 
            check_user_interrupt
        );
    };

    solver::solve_core(
        state,
        display,
        pb_add_suffix_f,
        update_loss_null_f,
        update_invariance_f,
        update_solutions_f,
        early_exit_f,
        screen_f,
        fit_f
    );
}

} // namespace cov
} // namespace gaussian
} // namespace solver
} // namespace adelie_core