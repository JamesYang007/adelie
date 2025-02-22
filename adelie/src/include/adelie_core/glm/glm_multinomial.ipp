#pragma once
#include <adelie_core/glm/glm_multinomial.hpp>

namespace adelie_core {
namespace glm {

ADELIE_CORE_GLM_MULTINOMIAL_TP
ADELIE_CORE_GLM_MULTINOMIAL::GlmMultinomial(
    const Eigen::Ref<const rowarr_value_t>& y,
    const Eigen::Ref<const vec_value_t>& weights
):
    base_t("multinomial", y, weights),
    _buff(y.rows() * (y.cols() + 1))
{
    if (y.cols() <= 1) {
        throw util::adelie_core_error(
            "y must have at least 2 columns (classes)."
        );
    }
}

ADELIE_CORE_GLM_MULTINOMIAL_TP
void
ADELIE_CORE_GLM_MULTINOMIAL::gradient(
    const Eigen::Ref<const rowarr_value_t>& eta,
    Eigen::Ref<rowarr_value_t> grad
)
{
    base_t::check_gradient(eta, grad);
    Eigen::Map<vec_value_t> eta_max(_buff.data(), y.rows());
    eta_max = eta.rowwise().maxCoeff();
    grad = (eta.colwise() - eta_max.matrix().transpose().array()).exp();
    auto& sum_exp = eta_max;
    sum_exp = grad.rowwise().sum();
    grad = (
        (y - grad.colwise() / sum_exp.matrix().transpose().array()).colwise() * 
        weights.matrix().transpose().array() / eta.cols()
    );
}

ADELIE_CORE_GLM_MULTINOMIAL_TP
void
ADELIE_CORE_GLM_MULTINOMIAL::hessian(
    const Eigen::Ref<const rowarr_value_t>& eta,
    const Eigen::Ref<const rowarr_value_t>& grad,
    Eigen::Ref<rowarr_value_t> hess
)
{
    base_t::check_hessian(eta, grad, hess);
        // K^{-1} W[:, None] * P
    hess = (
        y.colwise() * weights.matrix().transpose().array() / eta.cols() 
        - grad
    );
    // 2 * K^{-1} W[:, None] * P * (1 - P)
    hess *= 2 * (1 - grad.cols() * (
            hess.colwise() /
            (weights + (weights <= 0).template cast<value_t>()).matrix().transpose().array()
        )
    );
}

ADELIE_CORE_GLM_MULTINOMIAL_TP
typename ADELIE_CORE_GLM_MULTINOMIAL::value_t
ADELIE_CORE_GLM_MULTINOMIAL::loss(
    const Eigen::Ref<const rowarr_value_t>& eta
)
{
    base_t::check_loss(eta);
    Eigen::Map<vec_value_t> eta_max(_buff.data(), y.rows());
    eta_max = eta.rowwise().maxCoeff();
    Eigen::Map<rowarr_value_t> eta_shift(_buff.data() + y.rows(), y.rows(), y.cols());
    eta_shift = (eta.colwise() - eta_max.matrix().transpose().array());
    return (
        weights.matrix().transpose().array() * (
            - (y * eta_shift).rowwise().sum()
            + eta_shift.exp().rowwise().sum().log()
        )
    ).sum() / y.cols();
}

ADELIE_CORE_GLM_MULTINOMIAL_TP
typename ADELIE_CORE_GLM_MULTINOMIAL::value_t
ADELIE_CORE_GLM_MULTINOMIAL::loss_full()
{
    value_t loss = 0;
    for (int i = 0; i < y.rows(); ++i) {
        value_t sum = 0;
        for (int k = 0; k < y.cols(); ++k) {
            const auto log_yik = std::log(y(i,k));
            if (!(std::isinf(log_yik) || std::isnan(log_yik))) {
                sum += y(i,k) * log_yik;
            }
        }
        loss -= sum * weights[i] / y.cols();
    }    
    return loss;
}

ADELIE_CORE_GLM_MULTINOMIAL_TP
void
ADELIE_CORE_GLM_MULTINOMIAL::inv_link(
    const Eigen::Ref<const rowarr_value_t>& eta,
    Eigen::Ref<rowarr_value_t> out
)
{
    Eigen::Map<vec_value_t> eta_max(_buff.data(), y.rows());
    eta_max = eta.rowwise().maxCoeff();
    out = (eta.colwise() - eta_max.matrix().transpose().array()).exp();
    auto& sum_exp = eta_max;
    sum_exp = out.rowwise().sum();
    out.colwise() /= sum_exp.matrix().transpose().array();
}

} // namespace glm
} // namespace adelie_core