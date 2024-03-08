#pragma once
#include <adelie_core/glm/glm_multibase.hpp>

namespace adelie_core {
namespace glm {

template <class ValueType>
class GlmMultiGaussian: public GlmMultiBase<ValueType>
{
public:
    using base_t = GlmMultiBase<ValueType>;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using typename base_t::rowarr_value_t;
    using base_t::y;
    using base_t::weights;

    explicit GlmMultiGaussian(
        const Eigen::Ref<const rowarr_value_t>& y,
        const Eigen::Ref<const vec_value_t>& weights
    ):
        base_t("multigaussian", y, weights, false)
    {}

    void gradient(
        const Eigen::Ref<const rowarr_value_t>& eta,
        Eigen::Ref<rowarr_value_t> grad
    ) override
    {
        base_t::check_gradient(eta, grad);
        grad = ((y-eta).colwise() * weights.matrix().transpose().array()) / eta.cols();
    }

    void hessian(
        const Eigen::Ref<const rowarr_value_t>& eta,
        const Eigen::Ref<const rowarr_value_t>& grad,
        Eigen::Ref<rowarr_value_t> hess
    ) override
    {
        base_t::check_hessian(eta, grad, hess);
        hess.colwise() = weights.matrix().transpose().array() / hess.cols();
    }

    value_t loss(
        const Eigen::Ref<const rowarr_value_t>& eta
    ) override
    {
        base_t::check_loss(eta);
        return (
            weights.matrix().transpose().array() * 
            (0.5 * eta.square() - y * eta).rowwise().sum()
        ).sum() / y.cols();
    }

    value_t loss_full() override
    {
        return -0.5 * (
            (y.square().colwise() * weights.matrix().transpose().array()).sum()
        ) / y.cols();
    }
};

} // namespace glm
} // namespace adelie_core