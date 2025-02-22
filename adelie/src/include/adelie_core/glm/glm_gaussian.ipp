#pragma once
#include <adelie_core/glm/glm_gaussian.hpp>

namespace adelie_core {
namespace glm {

ADELIE_CORE_GLM_GAUSSIAN_TP
ADELIE_CORE_GLM_GAUSSIAN::GlmGaussian(
    const Eigen::Ref<const vec_value_t>& y,
    const Eigen::Ref<const vec_value_t>& weights
):
    base_t("gaussian", y, weights)
{}

ADELIE_CORE_GLM_GAUSSIAN_TP
void
ADELIE_CORE_GLM_GAUSSIAN::gradient(
    const Eigen::Ref<const vec_value_t>& eta,
    Eigen::Ref<vec_value_t> grad
) 
{
    base_t::check_gradient(eta, grad);
    grad = weights * (y - eta);
}

ADELIE_CORE_GLM_GAUSSIAN_TP
void
ADELIE_CORE_GLM_GAUSSIAN::hessian(
    const Eigen::Ref<const vec_value_t>& eta,
    const Eigen::Ref<const vec_value_t>& grad,
    Eigen::Ref<vec_value_t> hess
) 
{
    base_t::check_hessian(eta, grad, hess);
    hess = weights;
}

ADELIE_CORE_GLM_GAUSSIAN_TP
typename ADELIE_CORE_GLM_GAUSSIAN::value_t
ADELIE_CORE_GLM_GAUSSIAN::loss(
    const Eigen::Ref<const vec_value_t>& eta
) 
{
    base_t::check_loss(eta);
    return (weights * (0.5 * eta.square() - y * eta)).sum();
}

ADELIE_CORE_GLM_GAUSSIAN_TP
typename ADELIE_CORE_GLM_GAUSSIAN::value_t
ADELIE_CORE_GLM_GAUSSIAN::loss_full() 
{
    return -0.5 * (y.square() * weights).sum();
}

ADELIE_CORE_GLM_GAUSSIAN_TP
void
ADELIE_CORE_GLM_GAUSSIAN::inv_link(
    const Eigen::Ref<const vec_value_t>& eta,
    Eigen::Ref<vec_value_t> out
)
{
    out = eta;
}

} // namespace glm
} // namespace adelie_core