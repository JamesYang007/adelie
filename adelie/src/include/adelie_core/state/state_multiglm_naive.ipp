#pragma once
#include <adelie_core/solver/solver_multiglm_naive.hpp>
#include <adelie_core/state/state_multiglm_naive.hpp>

namespace adelie_core {
namespace state {

ADELIE_CORE_STATE_MULTI_GLM_NAIVE_TP
void
ADELIE_CORE_STATE_MULTI_GLM_NAIVE::solve(
    glm_t& glm,
    util::tq::progress_bar_t& pb,
    std::function<bool()> exit_cond,
    std::function<void()> check_user_interrupt
)
{
    solver::multiglm::naive::solve(*this, glm, pb, exit_cond, check_user_interrupt);
}

} // namespace state
} // namespace adelie_core