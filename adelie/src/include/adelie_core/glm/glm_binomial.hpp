#pragma once
#include <adelie_core/glm/glm_base.hpp>

namespace adelie_core {
namespace glm {

template <class ValueType>
class GlmBinomial: public GlmBase<ValueType>
{
public:
    using base_t = GlmBase<ValueType>;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using base_t::y;
    using base_t::weights;

    explicit GlmBinomial(
        const Eigen::Ref<const vec_value_t>& y,
        const Eigen::Ref<const vec_value_t>& weights
    ):
        base_t("binomial", y, weights)
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
        // numerically stable so that exp == 0 when y = 1 and eta = inf or y = 0 and eta = -inf.
        // NOTE: this only works when y takes values in {0,1}
        return (weights * (1 + ((1-2*y) * eta).exp()).log()).sum();
    }

    value_t loss_full() override
    {
        return 0;
    }
};

} // namespace glm
} // namespace adelie_core