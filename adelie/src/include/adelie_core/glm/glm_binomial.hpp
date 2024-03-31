#pragma once
#include <unsupported/Eigen/SpecialFunctions>
#include <adelie_core/glm/glm_base.hpp>
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
    using value_t = std::decay_t<YType>::Scalar;

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

template <class ValueType>
class GlmBinomialLogit: public GlmBase<ValueType>
{
public:
    using base_t = GlmBase<ValueType>;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using base_t::y;
    using base_t::weights;

    explicit GlmBinomialLogit(
        const Eigen::Ref<const vec_value_t>& y,
        const Eigen::Ref<const vec_value_t>& weights
    ):
        base_t("binomial_logit", y, weights)
    {}

    void gradient(
        const Eigen::Ref<const vec_value_t>& eta,
        Eigen::Ref<vec_value_t> grad
    ) override
    {
        base_t::check_gradient(eta, grad);
        grad = weights * (y - 1 / (1 + (-eta).exp()));
    }

    void hessian(
        const Eigen::Ref<const vec_value_t>& eta,
        const Eigen::Ref<const vec_value_t>& grad,
        Eigen::Ref<vec_value_t> hess
    ) override
    {
        base_t::check_hessian(eta, grad, hess);
        hess = weights * y - grad; // W * p
        // denominator is given a dummy positive value when weights == 0.
        hess = (hess * (weights-hess)) / (weights + (weights <= 0).template cast<value_t>());
    }

    value_t loss(
        const Eigen::Ref<const vec_value_t>& eta
    ) override
    {
        base_t::check_loss(eta);
        constexpr auto max = std::numeric_limits<value_t>::max();
        return (weights * (
            ((eta > 0).template cast<value_t>() - y) * eta.min(max).max(-max) +
            (1 + (-eta.abs()).exp()).log()
        )).sum();
    }

    value_t loss_full() override
    {
        return binomial::loss_full(y, weights);
    }
};

template <class ValueType>
class GlmBinomialProbit: public GlmBase<ValueType>
{
public:
    using base_t = GlmBase<ValueType>;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using base_t::y;
    using base_t::weights;

private:
    static auto std_cdf(
        const Eigen::Ref<const vec_value_t>& x
    )
    {
        constexpr value_t sqrt_2 = 1.4142135623730950488016887242096980785696718753769480731766797379;
        return 0.5 * (1 + (x / sqrt_2).erf());
    }

    static auto std_pdf(
        const Eigen::Ref<const vec_value_t>& x
    )
    {
        constexpr value_t sqrt_2pi_inv = 0.3989422804014326779399460599343818684758586311649346576659258296;
        return sqrt_2pi_inv * (-0.5 * x.square()).exp();
    }

    vec_value_t _buff;

public:
    explicit GlmBinomialProbit(
        const Eigen::Ref<const vec_value_t>& y,
        const Eigen::Ref<const vec_value_t>& weights
    ):
        base_t("binomial_probit", y, weights),
        _buff(y.size())
    {}

    void gradient(
        const Eigen::Ref<const vec_value_t>& eta,
        Eigen::Ref<vec_value_t> grad
    ) override
    {
        base_t::check_gradient(eta, grad);
        constexpr auto max = std::numeric_limits<value_t>::max();
        grad = std_cdf(eta);
        grad = weights * std_pdf(eta) * (
            y * (1 / grad).min(max) - (1-y) * (1 / (1-grad)).min(max)
        ); 
    }

    void hessian(
        const Eigen::Ref<const vec_value_t>& eta,
        const Eigen::Ref<const vec_value_t>& grad,
        Eigen::Ref<vec_value_t> hess
    ) override
    {
        base_t::check_hessian(eta, grad, hess);
        constexpr auto max = std::numeric_limits<value_t>::max();
        hess = std_cdf(eta);
        hess = weights * (
            y * (1 / hess.square()).min(max) + (1-y) * (1 / (1-hess).square()).min(max)
        ) * std_pdf(eta).square() + eta * grad;
    }

    value_t loss(
        const Eigen::Ref<const vec_value_t>& eta
    ) override
    {
        base_t::check_loss(eta);
        constexpr auto max = std::numeric_limits<value_t>::max();
        _buff = std_cdf(eta);
        return -(weights * (
            y * _buff.log().max(-max) + (1-y) * (1-_buff).log().max(-max)
        )).sum();
    }

    value_t loss_full() override
    {
        return binomial::loss_full(y, weights);
    }
};

} // namespace glm
} // namespace adelie_core