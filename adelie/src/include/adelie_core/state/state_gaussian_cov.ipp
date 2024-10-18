#pragma once
#include <adelie_core/solver/solver_gaussian_cov.hpp>
#include <adelie_core/state/state_gaussian_cov.hpp>

namespace adelie_core {
namespace state {

ADELIE_CORE_STATE_GAUSSIAN_COV_TP
void
ADELIE_CORE_STATE_GAUSSIAN_COV::initialize()
{
    if (v.size() != A->cols()) {
        throw util::adelie_core_error("v must be (p,) where A is (p, p).");
    }
    /* initialize the rest of the screen quantities */
    solver::gaussian::cov::update_screen_derived(*this);
}

ADELIE_CORE_STATE_GAUSSIAN_COV_TP
void
ADELIE_CORE_STATE_GAUSSIAN_COV::solve(
    util::tq::progress_bar_t& pb,
    std::function<bool()> exit_cond,
    std::function<void()> check_user_interrupt
)
{
    solver::gaussian::cov::solve(*this, pb, exit_cond, check_user_interrupt);
}

} // namespace state
} // namespace adelie_core