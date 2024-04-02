#pragma once
#include <adelie_core/state/state_gaussian_naive.hpp>
#include <adelie_core/state/state_multigaussian_naive.hpp>
#include <adelie_core/solver/solver_gaussian_naive.hpp>
#include <adelie_core/util/functional.hpp>
#include <memory>

namespace adelie_core {
namespace solver {
namespace multigaussian {
namespace naive {

template <class StateType,
          class ExitCondType,
          class UpdateCoefficientsType,
          class CUIType=util::no_op>
inline void solve(
    StateType&& state,
    bool display,
    ExitCondType exit_cond_f,
    UpdateCoefficientsType update_coefficients_f,
    CUIType check_user_interrupt = CUIType()
)
{
    using state_t = std::decay_t<StateType>;
    using vec_value_t = typename state_t::vec_value_t;
    using state_gaussian_naive_t = typename state_t::base_t;

    const auto n_classes = state.n_classes;
    const auto multi_intercept = state.multi_intercept;
    auto& betas = state.betas;
    auto& intercepts = state.intercepts;

    const auto tidy = [&]() {
        if (multi_intercept) {
            auto& beta = betas.back();
            intercepts.push_back(
                Eigen::Map<const vec_value_t>(beta.valuePtr(), n_classes)
            );
            beta = beta.tail(beta.size() - n_classes);
        } else {
            intercepts.push_back(
                vec_value_t::Zero(n_classes)
            );
        }
    };

    gaussian::naive::solve(
        static_cast<state_gaussian_naive_t&>(state),
        display,
        exit_cond_f,
        update_coefficients_f,
        tidy,
        check_user_interrupt
    );
}

} // namespace naive
} // namespace multigaussian
} // namespace solver
} // namespace adelie_core