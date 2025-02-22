#pragma once
#include <adelie_core/glm/glm_multigaussian.hpp>

namespace adelie_core {
namespace glm {

ADELIE_CORE_GLM_MULTIGAUSSIAN_TP
ADELIE_CORE_GLM_MULTIGAUSSIAN::GlmMultiGaussian(
    const Eigen::Ref<const rowarr_value_t>& y,
    const Eigen::Ref<const vec_value_t>& weights
):
    base_t("multigaussian", y, weights)
{}

ADELIE_CORE_GLM_MULTIGAUSSIAN_TP
void
ADELIE_CORE_GLM_MULTIGAUSSIAN::gradient(
    const Eigen::Ref<const rowarr_value_t>& eta,
    Eigen::Ref<rowarr_value_t> grad
)
{
    base_t::check_gradient(eta, grad);
    grad = ((y-eta).colwise() * weights.matrix().transpose().array()) / eta.cols();
}

ADELIE_CORE_GLM_MULTIGAUSSIAN_TP
void
ADELIE_CORE_GLM_MULTIGAUSSIAN::hessian(
    const Eigen::Ref<const rowarr_value_t>& eta,
    const Eigen::Ref<const rowarr_value_t>& grad,
    Eigen::Ref<rowarr_value_t> hess
)
{
    base_t::check_hessian(eta, grad, hess);
    hess.colwise() = weights.matrix().transpose().array() / hess.cols();
}

ADELIE_CORE_GLM_MULTIGAUSSIAN_TP
typename ADELIE_CORE_GLM_MULTIGAUSSIAN::value_t
ADELIE_CORE_GLM_MULTIGAUSSIAN::loss(
    const Eigen::Ref<const rowarr_value_t>& eta
)
{
    base_t::check_loss(eta);
    return (
        weights.matrix().transpose().array() * 
        (0.5 * eta.square() - y * eta).rowwise().sum()
    ).sum() / y.cols();
}

ADELIE_CORE_GLM_MULTIGAUSSIAN_TP
typename ADELIE_CORE_GLM_MULTIGAUSSIAN::value_t
ADELIE_CORE_GLM_MULTIGAUSSIAN::loss_full()
{
    return -0.5 * (
        (y.square().colwise() * weights.matrix().transpose().array()).sum()
    ) / y.cols();
}

ADELIE_CORE_GLM_MULTIGAUSSIAN_TP
void
ADELIE_CORE_GLM_MULTIGAUSSIAN::inv_link(
    const Eigen::Ref<const rowarr_value_t>& eta,
    Eigen::Ref<rowarr_value_t> out
)
{
    out = eta;
}

} // namespace glm
} // namespace adelie_core