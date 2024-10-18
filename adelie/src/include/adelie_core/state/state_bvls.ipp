#pragma once
#include <adelie_core/solver/solver_bvls.hpp>
#include <adelie_core/state/state_bvls.hpp>

namespace adelie_core {
namespace state {

ADELIE_CORE_STATE_BVLS_TP
void
ADELIE_CORE_STATE_BVLS::solve(
    std::function<void()> check_user_interrupt
)
{
    solver::bvls::solve(*this, check_user_interrupt);
}

} // namespace state
} // namespace adelie_core