#pragma once
#include <adelie_core/state/state_gaussian_cov.hpp>
#include <adelie_core/solver/solver_gaussian_cov.hpp>

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

} // namespace state
} // namespace adelie_core