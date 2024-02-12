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

    explicit GlmMultiGaussian():
        base_t("multigaussian", false)
    {}

    void gradient(
        const Eigen::Ref<const rowarr_value_t>& eta,
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<rowarr_value_t> mu
    ) override
    {
        mu = (eta.colwise() * weights.matrix().transpose().array()) / eta.cols();
    }

    void hessian(
        const Eigen::Ref<const rowarr_value_t>&,
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<rowarr_value_t> var
    ) override
    {
        var.colwise() = weights.matrix().transpose().array() / var.cols();
    }

    value_t deviance(
        const Eigen::Ref<const rowarr_value_t>& y,
        const Eigen::Ref<const rowarr_value_t>& eta,
        const Eigen::Ref<const vec_value_t>& weights
    ) override
    {
        return (
            weights.matrix().transpose().array() * 
            (0.5 * eta.square() - y * eta).rowwise().sum()
        ).sum() / y.cols();
    }

    value_t deviance_full(
        const Eigen::Ref<const rowarr_value_t>& y,
        const Eigen::Ref<const vec_value_t>& weights
    ) override
    {
        return -0.5 * (
            (y.square().colwise() * weights.matrix().transpose().array()).sum()
        ) / y.cols();
    }
};

} // namespace glm
} // namespace adelie_core