#pragma once
#include <adelie_core/solver/solver_gaussian_pin_cov.hpp>
#include <adelie_core/state/state_gaussian_pin_cov.hpp>

namespace adelie_core {
namespace state {

ADELIE_CORE_STATE_GAUSSIAN_PIN_COV_TP
void
ADELIE_CORE_STATE_GAUSSIAN_PIN_COV::initialize()
{
    // optimization
    active_subset_order.reserve(screen_subset_order.size());
    active_subset_ordered.reserve(screen_subset_order.size());
    inactive_subset_order.reserve(screen_subset_order.size());
    inactive_subset_ordered.reserve(screen_subset_order.size());

    solver::gaussian::pin::cov::update_active_inactive_subset(*this);
}

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