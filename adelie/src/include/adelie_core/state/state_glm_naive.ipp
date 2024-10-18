#pragma once
#include <adelie_core/solver/solver_glm_naive.hpp>
#include <adelie_core/state/state_glm_naive.hpp>

namespace adelie_core {
namespace state {

ADELIE_CORE_STATE_GLM_NAIVE_TP
void
ADELIE_CORE_STATE_GLM_NAIVE::solve(
    glm_t& glm,
    util::tq::progress_bar_t& pb,
    std::function<bool()> exit_cond,
    std::function<void()> check_user_interrupt
)
{
    solver::glm::naive::solve(*this, glm, pb, exit_cond, check_user_interrupt);
}

} // namespace state
} // namespace adelie_core