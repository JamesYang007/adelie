#pragma once
#include <adelie_core/state/state_gaussian_pin_cov.hpp>

namespace adelie_core {
namespace state {

ADELIE_CORE_STATE_GAUSSIAN_PIN_COV_TP
void
ADELIE_CORE_STATE_GAUSSIAN_PIN_COV::solve(
    std::function<void()> check_user_interrupt
)
{
    solver::gaussian::pin::cov::solve(*this, check_user_interrupt);
}

} // namespace state
} // namespace adelie_core