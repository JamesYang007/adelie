#pragma once
#include <adelie_core/state/state_gaussian_naive.hpp>

namespace adelie_core {
namespace state {

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