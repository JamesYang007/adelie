#pragma once
#include <adelie_core/solver/solver_pinball.hpp>
#include <adelie_core/state/state_pinball.hpp>

namespace adelie_core {
namespace state {

ADELIE_CORE_STATE_PINBALL_TP
void
ADELIE_CORE_STATE_PINBALL::initialize()
{
    const auto m = A->rows();
    const auto d = A->cols();

    if (S.rows() != d || S.cols() != d) {
        throw util::adelie_core_solver_error(
            "S must be (d, d) where A is (m, d). "
        );
    }
    if (penalty_neg.size() != m) {
        throw util::adelie_core_solver_error(
            "penalty_neg must be (m,) where A is (m, d). "
        );
    }
    if (penalty_pos.size() != m) {
        throw util::adelie_core_solver_error(
            "penalty_pos must be (m,) where A is (m, d). "
        );
    }
    if (kappa <= 0) {
        throw util::adelie_core_solver_error(
            "kappa must be > 0. "
        );
    }
    if (tol < 0) {
        throw util::adelie_core_solver_error(
            "tol must be >= 0."
        );
    }
    if (static_cast<Eigen::Index>(screen_set_size) > m) {
        throw util::adelie_core_solver_error(
            "screen_set_size must be <= m where A is (m, d). "
        );
    }
    if (screen_set.size() != m) {
        throw util::adelie_core_solver_error(
            "screen_set must be (m,) where A is (m, d). "
        );
    }
    if (is_screen.size() != m) {
        throw util::adelie_core_solver_error(
            "is_screen must be (m,) where A is (m, d). "
        );
    }
    if (screen_ASAT_diag.size() != m) {
        throw util::adelie_core_solver_error(
            "screen_ASAT_diag must be (m,) where A is (m, d). "
        );
    }
    if (screen_AS.rows() != m || screen_AS.cols() != d) {
        throw util::adelie_core_solver_error(
            "screen_AS must be (m, d) where A is (m, d). "
        );
    }
    if (static_cast<Eigen::Index>(active_set_size) > m) {
        throw util::adelie_core_solver_error(
            "active_set_size must be <= m where A is (m, d). "
        );
    }
    if (active_set.size() != m) {
        throw util::adelie_core_solver_error(
            "active_set must be (m,) where A is (m, d). "
        );
    }
    if (is_active.size() != m) {
        throw util::adelie_core_solver_error(
            "is_active must be (m,) where A is (m, d). "
        );
    }
    if (beta.size() != m) {
        throw util::adelie_core_solver_error(
            "beta must be (m,) where A is (m, d). "
        );
    }
    if (resid.size() != d) {
        throw util::adelie_core_solver_error(
            "resid must be (d,) where A is (m, d). "
        );
    }
    if (grad.size() != m) {
        throw util::adelie_core_solver_error(
            "grad must be (m,) where A is (m, d). "
        );
    }
}

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