#pragma once
#include <adelie_core/configs.hpp>
#include <adelie_core/glm/glm_base.hpp>

namespace adelie_core {
namespace glm {

ADELIE_CORE_GLM_BASE_TP
ADELIE_CORE_GLM_BASE::GlmBase(
    const string_t& name,
    const Eigen::Ref<const vec_value_t>& y,
    const Eigen::Ref<const vec_value_t>& weights
):
    name(name),
    y(y.data(), y.size()),
    weights(weights.data(), weights.size())
{
    if (y.size() != weights.size()) {
        throw util::adelie_core_error("y must be (n,) where weights is (n,).");
    }
}

ADELIE_CORE_GLM_BASE_TP
void
ADELIE_CORE_GLM_BASE::inv_hessian_gradient(
    const Eigen::Ref<const vec_value_t>& eta,
    const Eigen::Ref<const vec_value_t>& grad,
    const Eigen::Ref<const vec_value_t>& hess,
    Eigen::Ref<vec_value_t> inv_hess_grad
)
{
    check_inv_hessian_gradient(eta, grad, hess, inv_hess_grad);
    inv_hess_grad = grad / (
        hess.max(0) + 
        value_t(Configs::hessian_min) * (hess <= 0).template cast<value_t>()
    );
}

} // namespace glm
} // namespace adelie_core