#pragma once
#include <adelie_core/solver/solver_gaussian_naive.hpp>
#include <adelie_core/state/state_gaussian_naive.hpp>

namespace adelie_core {
namespace state {

ADELIE_CORE_STATE_GAUSSIAN_NAIVE_TP
void
ADELIE_CORE_STATE_GAUSSIAN_NAIVE::initialize()
{ 
    const auto n = X->rows();
    const auto p = X->cols();
    if (weights.size() != n) {
        throw util::adelie_core_error("weights must be (n,) where X is (n, p).");
    }
    if (X_means.size() != p) {
        throw util::adelie_core_error("X_means must be (p,) where X is (n, p).");
    }
    if (resid.size() != n) {
        throw util::adelie_core_error("resid must be (n,) where X is (n, p).");
    }
    if (this->grad.size() != p) {
        throw util::adelie_core_error("grad must be (p,) where X is (n, p).");
    }
    /* initialize the rest of the screen quantities */
    solver::gaussian::naive::update_screen_derived(*this); 
}

ADELIE_CORE_STATE_GAUSSIAN_NAIVE_TP
void
ADELIE_CORE_STATE_GAUSSIAN_NAIVE::solve(
    util::tq::progress_bar_t& pb,
    std::function<bool()> exit_cond,
    std::function<void()> check_user_interrupt
)
{
    solver::gaussian::naive::solve(*this, pb, exit_cond, check_user_interrupt);
}

} // namespace state
} // namespace adelie_core