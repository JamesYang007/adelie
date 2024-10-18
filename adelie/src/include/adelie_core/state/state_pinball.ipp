#pragma once
#include <adelie_core/solver/solver_pinball.hpp>
#include <adelie_core/state/state_pinball.hpp>

namespace adelie_core {
namespace state {

ADELIE_CORE_STATE_PINBALL_TP
void
ADELIE_CORE_STATE_PINBALL::solve(
    std::function<void()> check_user_interrupt
)
{
    solver::pinball::solve(*this, check_user_interrupt);
}

} // namespace state
} // namespace adelie_core