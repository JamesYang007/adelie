#pragma once
#include <adelie_core/solver/solver_css_cov.hpp>
#include <adelie_core/state/state_css_cov.hpp>

namespace adelie_core {
namespace state {

ADELIE_CORE_STATE_CSS_COV_TP
void
ADELIE_CORE_STATE_CSS_COV::initialize()
{
    const auto n = S.rows();
    const auto p = S.cols();

    if (n != p) {
        throw util::adelie_core_error(
            "S must be (p, p)."
        );
    }
    if (subset_size > static_cast<size_t>(p)) {
        throw util::adelie_core_error(
            "subset_size must be <= p."
        );
    }
    if (
        method == util::css_method_type::_swapping && 
        subset_size != subset.size()
    ) {
        throw util::adelie_core_error(
            "subset must be (subset_size,) if method is \"swapping\"."
        );
    }
    if (
        method == util::css_method_type::_swapping &&
        subset.size() &&
        (
            Eigen::Map<vec_index_t>(subset.data(), subset.size()).minCoeff() < 0 ||
            Eigen::Map<vec_index_t>(subset.data(), subset.size()).maxCoeff() >= p
        )
    ) {
        throw util::adelie_core_error(
            "subset must be in the range [0, p)."
        );
    }
    if (
        method == util::css_method_type::_greedy &&
        subset.size()
    ) {
        throw util::adelie_core_error(
            "subset must be empty if method is \"greedy\"."
        );
    }
    if (n_threads < 1) {
        throw util::adelie_core_error(
            "n_threads must be >= 1."
        );
    }
}

ADELIE_CORE_STATE_CSS_COV_TP
void
ADELIE_CORE_STATE_CSS_COV::solve(
    std::function<void()> check_user_interrupt
)
{
    solver::css::cov::solve(*this, check_user_interrupt);
}

} // namespace state
} // namespace adelie_core