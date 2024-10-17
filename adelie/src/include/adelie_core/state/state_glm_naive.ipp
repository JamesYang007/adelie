#pragma once
#include <adelie_core/state/state_glm_naive.hpp>

namespace adelie_core {
namespace state {

ADELIE_CORE_STATE_GLM_NAIVE_TP
void
ADELIE_CORE_STATE_GLM_NAIVE::initialize()
{
    const auto n = X->rows();
    if (offsets.size() != n) {
        throw util::adelie_core_error("offsets must be (n,) where X is (n, p).");
    }
    if (eta.size() != n) {
        throw util::adelie_core_error("eta must be (n,) where X is (n, p).");
    }
    if (resid.size() != n) {
        throw util::adelie_core_error("resid must be (n,) where X is (n, p).");
    }
    if (irls_tol <= 0) {
        throw util::adelie_core_error("irls_tol must be > 0.");
    }
}

} // namespace state
} // namespace adelie_core