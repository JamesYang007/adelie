#pragma once
#include <adelie_core/util/exceptions.hpp>
#include <adelie_core/util/functional.hpp>
#include <adelie_core/util/macros.hpp>
#include <adelie_core/util/stopwatch.hpp>

namespace adelie_core {
namespace solver {
namespace pinball {

template <
    class StateType, 
    class Iter,
    class ValueType,
    class AdditionalStepType=util::no_op
>
ADELIE_CORE_STRONG_INLINE
void coordinate_descent(
    StateType&& state,
    Iter begin,
    Iter end,
    ValueType& convg_measure,
    AdditionalStepType additional_step=AdditionalStepType()
)
{
    using state_t = std::decay_t<StateType>;
    using value_t = typename state_t::value_t;
    auto& A = *state.A;
    const auto& penalty_neg = state.penalty_neg;
    const auto& penalty_pos = state.penalty_pos;
    const auto& screen_ASAT_diag = state.screen_ASAT_diag;
    const auto& screen_AS = state.screen_AS;
    auto& beta = state.beta;
    auto& resid = state.resid;
    auto& loss = state.loss;

    for (auto it = begin; it != end; ++it) {
        const auto k = *it;
        const auto vk = screen_ASAT_diag[k];
        const auto lk = penalty_neg[k];
        const auto uk = penalty_pos[k];
        const auto AS_k = screen_AS.row(k);
        const auto gk = A.rvmul(k, resid);
        auto& bk = beta[k];
        const auto bk_old = bk;
        const auto gk0 = gk + vk * bk_old;
        const auto gk0_lk = gk0 + lk;
        bk = (vk <= 0) ? bk_old : std::copysign(
            std::max<value_t>(std::max<value_t>(
                -gk0_lk, gk0-uk
            ), 0),
            gk0_lk 
        ) / vk;
        if (bk == bk_old) continue;
        const auto del = bk - bk_old;
        const auto scaled_del_sq = vk * del * del; 
        convg_measure = std::max<value_t>(convg_measure, scaled_del_sq);
        loss -= del * gk - 0.5 * scaled_del_sq;
        resid.matrix() -= del * AS_k;
        additional_step(k);
    }
}

template <
    class StateType, 
    class CheckUserInterruptType=util::no_op
>
inline void solve_active(
    StateType&& state,
    CheckUserInterruptType check_user_interrupt=CheckUserInterruptType()
)
{
    using state_t = std::decay_t<StateType>;
    using value_t = typename state_t::value_t;

    const auto y_var = state.y_var;
    const auto& active_set_size = state.active_set_size;
    const auto& active_set = state.active_set;
    const auto max_iters = state.max_iters;
    const auto tol = state.tol;
    auto& iters = state.iters;

    while (1) {
        check_user_interrupt();
        ++iters;
        value_t convg_measure = 0;
        coordinate_descent(
            state,
            active_set.data(),
            active_set.data() + active_set_size,
            convg_measure
        );
        if (iters >= max_iters) {
            throw util::adelie_core_solver_error(
                "pinball: max iterations reached!"
            );
        }
        if (convg_measure <= tol * y_var) break;
    }
}

template <
    class StateType, 
    class CheckUserInterruptType=util::no_op
>
inline void fit(
    StateType&& state,
    CheckUserInterruptType check_user_interrupt=CheckUserInterruptType()
)
{
    using state_t = std::decay_t<StateType>;
    using value_t = typename state_t::value_t;

    const auto y_var = state.y_var;
    const auto& screen_set_size = state.screen_set_size;
    const auto& screen_set = state.screen_set;
    const auto max_iters = state.max_iters;
    const auto tol = state.tol;
    const auto& beta = state.beta;
    auto& active_set_size = state.active_set_size;
    auto& active_set = state.active_set;
    auto& is_active = state.is_active;
    auto& iters = state.iters;

    #ifdef ADELIE_CORE_DEBUG
    util::Stopwatch sw;
    double fit_active_time = 0;
    double fit_screen_time = 0;
    #endif

    const auto add_active = [&](auto k) {
        if (!is_active[k]) {
            active_set[active_set_size] = k;
            is_active[k] = true;
            ++active_set_size;
        }
    };
    const auto prune = [&]() {
        size_t n_active = 0;
        for (size_t i = 0; i < active_set_size; ++i) {
            const auto k = active_set[i];
            const auto bk = beta[k];
            if (bk == 0) {
                is_active[k] = false;
                continue;
            }
            active_set[n_active] = k;
            ++n_active;
        }
        active_set_size = n_active;
    };

    while (1) {
        check_user_interrupt();
        ++iters;
        value_t convg_measure = 0;
        #ifdef ADELIE_CORE_DEBUG
        sw.start();
        #endif
        coordinate_descent(
            state,
            screen_set.data(),
            screen_set.data() + screen_set_size,
            convg_measure,
            add_active
        );
        #ifdef ADELIE_CORE_DEBUG
        fit_screen_time += sw.elapsed();
        #endif
        if (iters >= max_iters) {
            throw util::adelie_core_solver_error(
                "pinball: max iterations reached!"
            );
        }
        if (convg_measure <= tol * y_var) {
            prune();
            break;
        }

        #ifdef ADELIE_CORE_DEBUG
        sw.start();
        #endif
        solve_active(state, check_user_interrupt);
        #ifdef ADELIE_CORE_DEBUG
        fit_active_time += sw.elapsed();
        #endif
        prune();
    }

    #ifdef ADELIE_CORE_DEBUG
    state.benchmark_fit_active.push_back(fit_active_time);
    state.benchmark_fit_screen.push_back(fit_screen_time);
    #endif
}

template <
    class StateType, 
    class ViolsOrderType
>
inline bool kkt_screen(
    StateType&& state,
    ViolsOrderType& viols_order
)
{
    using state_t = std::decay_t<StateType>;
    using value_t = typename state_t::value_t;

    auto& A = *state.A;
    const auto& S = state.S;
    const auto& penalty_neg = state.penalty_neg;
    const auto& penalty_pos = state.penalty_pos;
    const auto& resid = state.resid;
    const auto kappa = state.kappa;
    auto& screen_set_size = state.screen_set_size;
    auto& screen_set = state.screen_set;
    auto& is_screen = state.is_screen;
    auto& screen_ASAT_diag = state.screen_ASAT_diag;
    auto& screen_AS = state.screen_AS;
    auto& grad = state.grad;

    const auto m = grad.size();

    ++state.n_kkt;

    #ifdef ADELIE_CORE_DEBUG
    util::Stopwatch sw;
    #endif

    // compute gradient
    #ifdef ADELIE_CORE_DEBUG
    sw.start();
    #endif
    A.tmul(resid, grad);
    #ifdef ADELIE_CORE_DEBUG
    state.benchmark_gradient.push_back(sw.elapsed());
    #endif

    // compute violations
    auto& viols = grad;
    viols = (grad-penalty_pos).max(-penalty_neg-grad);

    // sort violations in decreasing order
    #ifdef ADELIE_CORE_DEBUG
    sw.start();
    #endif
    std::sort(
        viols_order.data(), 
        viols_order.data() + m,
        [&](auto i, auto j) { return viols[i] > viols[j]; } 
    );
    #ifdef ADELIE_CORE_DEBUG
    state.benchmark_viols_sort.push_back(sw.elapsed());
    #endif

    const auto screen_set_size_old = screen_set_size;
    bool kkt_passed = true;

    // check KKT and screen
    for (Eigen::Index j = 0; j < m; ++j) {
        const auto k = viols_order[j];
        const auto vk = viols[k];
        if (is_screen[k] || vk <= 0) continue;
        kkt_passed = false;
        // break if reached max active capacity
        if (screen_set_size >= screen_set_size_old + kappa) break;
        // otherwise add violator to the screen set
        screen_set[screen_set_size] = k;
        is_screen[k] = true;
        auto AS_k = screen_AS.row(k);
        A.rmmul(k, S, AS_k);
        screen_ASAT_diag[k] = std::max<value_t>(A.rvmul(k, AS_k), 0);
        ++screen_set_size;
    }

    return kkt_passed;
}

template <
    class StateType, 
    class CheckUserInterruptType=util::no_op
>
inline void solve(
    StateType&& state,
    CheckUserInterruptType check_user_interrupt=CheckUserInterruptType()
)
{
    using state_t = std::decay_t<StateType>;
    using vec_index_t = typename state_t::vec_index_t; 

    const auto m = state.grad.size();
    vec_index_t viols_order = vec_index_t::LinSpaced(m, 0, m-1);

    while (1) {
        const auto loss_prev = state.loss;
        fit(state, check_user_interrupt);
        #ifdef ADELIE_CORE_DEBUG
        state.dbg_beta.push_back(state.beta);
        state.dbg_active_set.push_back(state.active_set.head(state.active_set_size));
        state.dbg_iter.push_back(state.iters);
        state.dbg_loss.push_back(state.loss);
        #endif
        if (
            state.n_kkt > 0 &&
            std::abs(state.loss-loss_prev) < 1e-6 * std::abs(state.y_var)
        ) return;
        const bool kkt_passed = kkt_screen(state, viols_order);
        if (kkt_passed) return;
    }
}

} // namespace pinball
} // namespace solver
} // namespace adelie_core