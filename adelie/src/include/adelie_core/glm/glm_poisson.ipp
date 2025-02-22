#pragma once
#include <adelie_core/glm/glm_poisson.hpp>

namespace adelie_core {
namespace glm {

ADELIE_CORE_GLM_POISSON_TP
ADELIE_CORE_GLM_POISSON::GlmPoisson(
    const Eigen::Ref<const vec_value_t>& y,
    const Eigen::Ref<const vec_value_t>& weights
):
    base_t("poisson", y, weights)
{}

ADELIE_CORE_GLM_POISSON_TP
void
ADELIE_CORE_GLM_POISSON::gradient(
    const Eigen::Ref<const vec_value_t>& eta,
    Eigen::Ref<vec_value_t> grad
) 
{
    base_t::check_gradient(eta, grad);
    grad = weights * (y - eta.exp());
}

ADELIE_CORE_GLM_POISSON_TP
void
ADELIE_CORE_GLM_POISSON::hessian(
    const Eigen::Ref<const vec_value_t>& eta,
    const Eigen::Ref<const vec_value_t>& grad,
    Eigen::Ref<vec_value_t> hess
) 
{
    base_t::check_hessian(eta, grad, hess);
    hess = weights * y - grad;
}

ADELIE_CORE_GLM_POISSON_TP
typename ADELIE_CORE_GLM_POISSON::value_t
ADELIE_CORE_GLM_POISSON::loss(
    const Eigen::Ref<const vec_value_t>& eta
) 
{
    base_t::check_loss(eta);
    // numerically stable when y == 0 and eta could be -inf
    return (weights * ((-eta).min(std::numeric_limits<value_t>::max()) * y + eta.exp())).sum();
}

ADELIE_CORE_GLM_POISSON_TP
typename ADELIE_CORE_GLM_POISSON::value_t
ADELIE_CORE_GLM_POISSON::loss_full() 
{
    return (weights * ((-y.log()).min(std::numeric_limits<value_t>::max()) * y + y)).sum();
}

ADELIE_CORE_GLM_POISSON_TP
void
ADELIE_CORE_GLM_POISSON::inv_link(
    const Eigen::Ref<const vec_value_t>& eta,
    Eigen::Ref<vec_value_t> out
)
{
    out = eta.exp();
}

} // namespace glm
} // namespace adelie_core