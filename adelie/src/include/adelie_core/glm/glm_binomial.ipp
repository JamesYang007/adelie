#pragma once
#define _USE_MATH_DEFINES // for C++
#include <cmath>
#include <unsupported/Eigen/SpecialFunctions>
#include <adelie_core/glm/glm_binomial.hpp>
#include <adelie_core/util/macros.hpp>

namespace adelie_core {
namespace glm {
namespace binomial {

template <class YType, class WeightsType>
ADELIE_CORE_STRONG_INLINE
auto loss_full(
    const YType& y,
    const WeightsType& weights
)
{
    using value_t = typename std::decay_t<YType>::Scalar;

    value_t loss = 0;
    for (int i = 0; i < weights.size(); ++i) {
        const auto log_yi = std::log(y[i]);
        const auto log_1myi = std::log(1-y[i]);
        if (!(std::isinf(log_yi) || std::isnan(log_yi))) {
            loss -= weights[i] * y[i] * log_yi;
        }
        if (!(std::isinf(log_1myi) || std::isnan(log_1myi))) {
            loss -= weights[i] * (1-y[i]) * log_1myi;
        }
    }
    return loss;
}

} // namespace binomial

ADELIE_CORE_GLM_BINOMIAL_LOGIT_TP
ADELIE_CORE_GLM_BINOMIAL_LOGIT::GlmBinomialLogit(
    const Eigen::Ref<const vec_value_t>& y,
    const Eigen::Ref<const vec_value_t>& weights
):
    base_t("binomial_logit", y, weights)
{}

ADELIE_CORE_GLM_BINOMIAL_LOGIT_TP
void
ADELIE_CORE_GLM_BINOMIAL_LOGIT::gradient(
    const Eigen::Ref<const vec_value_t>& eta,
    Eigen::Ref<vec_value_t> grad
) 
{
    base_t::check_gradient(eta, grad);
    grad = weights * (y - 1 / (1 + (-eta).exp()));
}

ADELIE_CORE_GLM_BINOMIAL_LOGIT_TP
void
ADELIE_CORE_GLM_BINOMIAL_LOGIT::hessian(
    const Eigen::Ref<const vec_value_t>& eta,
    const Eigen::Ref<const vec_value_t>& grad,
    Eigen::Ref<vec_value_t> hess
) 
{
    base_t::check_hessian(eta, grad, hess);
    hess = weights * y - grad; // W * p
    // denominator is given a dummy positive value when weights == 0.
    hess = (hess * (weights-hess)) / (weights + (weights <= 0).template cast<value_t>());
}

ADELIE_CORE_GLM_BINOMIAL_LOGIT_TP
typename ADELIE_CORE_GLM_BINOMIAL_LOGIT::value_t
ADELIE_CORE_GLM_BINOMIAL_LOGIT::loss(
    const Eigen::Ref<const vec_value_t>& eta
) 
{
    base_t::check_loss(eta);
    constexpr auto max = std::numeric_limits<value_t>::max();
    return (weights * (
        ((eta > 0).template cast<value_t>() - y) * eta.min(max).max(-max) +
        (1 + (-eta.abs()).exp()).log()
    )).sum();
}

ADELIE_CORE_GLM_BINOMIAL_LOGIT_TP
typename ADELIE_CORE_GLM_BINOMIAL_LOGIT::value_t
ADELIE_CORE_GLM_BINOMIAL_LOGIT::loss_full() 
{
    return binomial::loss_full(y, weights);
}

ADELIE_CORE_GLM_BINOMIAL_LOGIT_TP
void
ADELIE_CORE_GLM_BINOMIAL_LOGIT::inv_link(
    const Eigen::Ref<const vec_value_t>& eta,
    Eigen::Ref<vec_value_t> out
)
{
    out = 1 / (1 + (-eta).exp());
}

ADELIE_CORE_GLM_BINOMIAL_PROBIT_TP
auto
ADELIE_CORE_GLM_BINOMIAL_PROBIT::std_cdf(
    const Eigen::Ref<const vec_value_t>& x
)
{
    constexpr value_t sqrt_2 = M_SQRT2;
    return 0.5 * (1 + (x / sqrt_2).erf());
}

ADELIE_CORE_GLM_BINOMIAL_PROBIT_TP
auto
ADELIE_CORE_GLM_BINOMIAL_PROBIT::std_pdf(
    const Eigen::Ref<const vec_value_t>& x
)
{
    constexpr value_t sqrt_2pi_inv = 0.5 * M_2_SQRTPI / M_SQRT2;
    return sqrt_2pi_inv * (-0.5 * x.square()).exp();
}

ADELIE_CORE_GLM_BINOMIAL_PROBIT_TP
ADELIE_CORE_GLM_BINOMIAL_PROBIT::GlmBinomialProbit(
    const Eigen::Ref<const vec_value_t>& y,
    const Eigen::Ref<const vec_value_t>& weights
):
    base_t("binomial_probit", y, weights),
    _buff(y.size())
{}

ADELIE_CORE_GLM_BINOMIAL_PROBIT_TP
void
ADELIE_CORE_GLM_BINOMIAL_PROBIT::gradient(
    const Eigen::Ref<const vec_value_t>& eta,
    Eigen::Ref<vec_value_t> grad
) 
{
    base_t::check_gradient(eta, grad);
    constexpr auto max = std::numeric_limits<value_t>::max();
    grad = std_cdf(eta);
    grad = weights * std_pdf(eta) * (
        y * (1 / grad).min(max) - (1-y) * (1 / (1-grad)).min(max)
    ); 
}

ADELIE_CORE_GLM_BINOMIAL_PROBIT_TP
void
ADELIE_CORE_GLM_BINOMIAL_PROBIT::hessian(
    const Eigen::Ref<const vec_value_t>& eta,
    const Eigen::Ref<const vec_value_t>& grad,
    Eigen::Ref<vec_value_t> hess
) 
{
    base_t::check_hessian(eta, grad, hess);
    constexpr auto max = std::numeric_limits<value_t>::max();
    hess = std_cdf(eta);
    hess = weights * (
        y * (1 / hess.square()).min(max) + (1-y) * (1 / (1-hess).square()).min(max)
    ) * std_pdf(eta).square() + eta * grad;
}

ADELIE_CORE_GLM_BINOMIAL_PROBIT_TP
typename ADELIE_CORE_GLM_BINOMIAL_PROBIT::value_t
ADELIE_CORE_GLM_BINOMIAL_PROBIT::loss(
    const Eigen::Ref<const vec_value_t>& eta
) 
{
    base_t::check_loss(eta);
    constexpr auto max = std::numeric_limits<value_t>::max();
    _buff = std_cdf(eta);
    return -(weights * (
        y * _buff.log().max(-max) + (1-y) * (1-_buff).log().max(-max)
    )).sum();
}

ADELIE_CORE_GLM_BINOMIAL_PROBIT_TP
typename ADELIE_CORE_GLM_BINOMIAL_PROBIT::value_t
ADELIE_CORE_GLM_BINOMIAL_PROBIT::loss_full() 
{
    return binomial::loss_full(y, weights);
}

ADELIE_CORE_GLM_BINOMIAL_PROBIT_TP
void
ADELIE_CORE_GLM_BINOMIAL_PROBIT::inv_link(
    const Eigen::Ref<const vec_value_t>& eta,
    Eigen::Ref<vec_value_t> out
)
{
    out = std_cdf(eta);
}

} // namespace glm
} // namespace adelie_core