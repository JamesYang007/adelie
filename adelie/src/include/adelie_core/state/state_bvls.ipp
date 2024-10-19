#pragma once
#include <adelie_core/solver/solver_bvls.hpp>
#include <adelie_core/state/state_bvls.hpp>

namespace adelie_core {
namespace state {

ADELIE_CORE_STATE_BVLS_TP
void
ADELIE_CORE_STATE_BVLS::initialize()
{
    const auto n = X->rows();
    const auto p = X->cols();

    if (X_vars.size() != p) {
        throw util::adelie_core_solver_error(
            "X_vars must be (p,) where X is (n, p). "
        );
    }
    if (lower.size() != p) {
        throw util::adelie_core_solver_error(
            "lower must be (p,) where X is (n, p). "
        );
    }
    if (upper.size() != p) {
        throw util::adelie_core_solver_error(
            "upper must be (p,) where X is (n, p). "
        );
    }
    if (weights.size() != n) {
        throw util::adelie_core_solver_error(
            "weights must be (n,) where X is (n, p). "
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
    if (static_cast<Eigen::Index>(active_set_size) > p) {
        throw util::adelie_core_solver_error(
            "active_set_size must be <= p where X is (n, p). "
        );
    }
    if (active_set.size() != p) {
        throw util::adelie_core_solver_error(
            "active_set must be (p,) where X is (n, p). "
        );
    }
    if (is_active.size() != p) {
        throw util::adelie_core_solver_error(
            "is_active must be (p,) where X is (n, p). "
        );
    }
    if (beta.size() != p) {
        throw util::adelie_core_solver_error(
            "beta must be (p,) where X is (p, n). "
        );
    }
    if (resid.size() != n) {
        throw util::adelie_core_solver_error(
            "resid must be (n,) where X is (n, p). "
        );
    }
    if (grad.size() != p) {
        throw util::adelie_core_solver_error(
            "grad must be (p,) where X is (n, p). "
        );
    }
}

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