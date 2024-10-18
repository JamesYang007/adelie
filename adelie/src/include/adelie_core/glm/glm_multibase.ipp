#pragma once
#include <adelie_core/configs.hpp>
#include <adelie_core/glm/glm_multibase.hpp>

namespace adelie_core {
namespace glm {

ADELIE_CORE_GLM_MULTIBASE_TP
ADELIE_CORE_GLM_MULTIBASE::GlmMultiBase(
    const string_t& name,
    const Eigen::Ref<const rowarr_value_t>& y,
    const Eigen::Ref<const vec_value_t>& weights
):
    name(name),
    y(y.data(), y.rows(), y.cols()),
    weights(weights.data(), weights.size())
{
    if (y.rows() != weights.size()) {
        throw util::adelie_core_error("y must be (n, K) where weights is (n,).");
    }
}

ADELIE_CORE_GLM_MULTIBASE_TP
void
ADELIE_CORE_GLM_MULTIBASE::inv_hessian_gradient(
    const Eigen::Ref<const rowarr_value_t>& eta,
    const Eigen::Ref<const rowarr_value_t>& grad,
    const Eigen::Ref<const rowarr_value_t>& hess,
    Eigen::Ref<rowarr_value_t> inv_hess_grad
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