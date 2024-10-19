#pragma once
#include <adelie_core/solver/solver_gaussian_pin_naive.hpp>
#include <adelie_core/state/state_gaussian_pin_naive.hpp>

namespace adelie_core {
namespace state {

ADELIE_CORE_STATE_GAUSSIAN_PIN_NAIVE_TP
void
ADELIE_CORE_STATE_GAUSSIAN_PIN_NAIVE::solve(
    std::function<void()> check_user_interrupt
)
{
    solver::gaussian::pin::naive::solve(*this, check_user_interrupt);
}

} // namespace state
} // namespace adelie_core