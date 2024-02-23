#pragma once
#include <adelie_core/glm/glm_base.hpp>

namespace adelie_core {
namespace glm {

template <class ValueType>
class GlmGaussian: public GlmBase<ValueType>
{
public:
    using base_t = GlmBase<ValueType>;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using base_t::y;
    using base_t::weights;

    explicit GlmGaussian(
        const Eigen::Ref<const vec_value_t>& y,
        const Eigen::Ref<const vec_value_t>& weights
    ):
        base_t("gaussian", y, weights)
    {}

    void gradient(
        const Eigen::Ref<const vec_value_t>& eta,
        Eigen::Ref<vec_value_t> grad
    ) override
    {
        base_t::check_gradient(eta, grad);
        grad = weights * (y - eta);
    }

    void hessian(
        const Eigen::Ref<const vec_value_t>& eta,
        const Eigen::Ref<const vec_value_t>& grad,
        Eigen::Ref<vec_value_t> hess
    ) override
    {
        base_t::check_hessian(eta, grad, hess);
        hess = weights;
    }

    value_t loss(
        const Eigen::Ref<const vec_value_t>& eta
    ) override
    {
        base_t::check_loss(eta);
        return (weights * (0.5 * eta.square() - y * eta)).sum();
    }

    value_t loss_full() override
    {
        return -0.5 * (y.square() * weights).sum();
    }
};

} // namespace glm
} // namespace adelie_core