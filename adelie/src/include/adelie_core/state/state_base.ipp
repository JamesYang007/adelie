#pragma once
#include <adelie_core/solver/solver_base.hpp>
#include <adelie_core/state/state_base.hpp>
#include <adelie_core/util/exceptions.hpp>

namespace adelie_core {
namespace state {

ADELIE_CORE_STATE_BASE_TP
void
ADELIE_CORE_STATE_BASE::initialize()
{
    // sanity checks
    const auto G = groups.size();
    if (constraints.size() != static_cast<size_t>(G)) {
        throw util::adelie_core_error("constraints must be (G,) where groups is (G,).");
    }
    {
        std::unordered_set<constraint_t*> constraints_set;
        for (auto c : constraints) {
            if (!c) continue;
            if (constraints_set.find(c) != constraints_set.end()) {
                throw util::adelie_core_error("constraints must contain distinct objects or nullptr.");
            }
            constraints_set.insert(c);
        }
    }
    if (group_sizes.size() != G) {
        throw util::adelie_core_error("group_sizes must be (G,) where groups is (G,).");
    }
    if (dual_groups.size() != G) {
        throw util::adelie_core_error("dual_groups must be (G,) where groups is (G,).");
    }
    if (penalty.size() != G) {
        throw util::adelie_core_error("penalty must be (G,) where groups is (G,).");
    }
    if (alpha < 0 || alpha > 1) {
        throw util::adelie_core_error("alpha must be in [0,1].");
    }
    if (tol < 0) {
        throw util::adelie_core_error("tol must be >= 0.");
    }
    if (adev_tol < 0 || adev_tol > 1) {
        throw util::adelie_core_error("adev_tol must be in [0,1].");
    }
    if (ddev_tol < 0 || ddev_tol > 1) {
        throw util::adelie_core_error("ddev_tol must be in [0,1].");
    }
    if (newton_tol < 0) {
        throw util::adelie_core_error("newton_tol must be >= 0.");
    }
    if (n_threads < 1) {
        throw util::adelie_core_error("n_threads must be >= 1.");
    }
    if (min_ratio < 0 || min_ratio > 1) {
        throw util::adelie_core_error("min_ratio must be in [0,1].");
    }
    if (pivot_subset_ratio <= 0 || pivot_subset_ratio > 1) {
        throw util::adelie_core_error("pivot_subset_ratio must be in (0,1].");
    }
    if (pivot_subset_min < 1) {
        throw util::adelie_core_error("pivot_subset_min must be >= 1.");
    }
    if (pivot_slack_ratio < 0) {
        throw util::adelie_core_error("pivot_slack_ratio must be >= 0.");
    }
    if (screen_set.size() != screen_is_active.size()) {
        throw util::adelie_core_error("screen_is_active must be (s,) where screen_set is (s,).");
    }
    if (screen_beta.size() < screen_set.size()) {
        throw util::adelie_core_error(
            "screen_beta must be (bs,) where bs >= s and screen_set is (s,). "
            "It is likely screen_beta has been initialized incorrectly. "
        );
    }
    if (active_set_size > static_cast<size_t>(G)) {
        throw util::adelie_core_error(
            "active_set_size must be <= G where groups is (G,)."
        );
    }
    if (active_set.size() != G) {
        throw util::adelie_core_error(
            "active_set must be (G,) where groups is (G,)."
        );
    }
    if (grad.size() != groups[G-1] + group_sizes[G-1]) {
        throw util::adelie_core_error(
            "grad.size() != groups[G-1] + group_sizes[G-1]. "
            "It is likely either grad has the wrong shape, "
            "or groups/group_sizes have been initialized incorrectly."
        );
    }

    /* initialize screen_set derived quantities */
    screen_begins.reserve(screen_set.size());
    solver::update_screen_derived_base(*this);

    /* initialize abs_grad */
    solver::update_abs_grad(*this, lmda);

    /* optimize for output storage size */
    const auto n_lmdas = std::max<size_t>(lmda_path.size(), lmda_path_size);
    betas.reserve(n_lmdas);
    duals.reserve(n_lmdas);
    intercepts.reserve(n_lmdas);
    devs.reserve(n_lmdas);
    lmdas.reserve(n_lmdas);
    benchmark_fit_screen.reserve(n_lmdas);
    benchmark_fit_active.reserve(n_lmdas);
    benchmark_kkt.reserve(n_lmdas);
    benchmark_screen.reserve(n_lmdas);
    benchmark_invariance.reserve(n_lmdas);
    n_valid_solutions.reserve(n_lmdas);
    active_sizes.reserve(n_lmdas);
    screen_sizes.reserve(n_lmdas);
}

} // namespace state
} // namespace adelie_core