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

    explicit GlmGaussian():
        base_t("gaussian", false)
    {}

    void gradient(
        const Eigen::Ref<const vec_value_t>& eta,
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> mu
    ) override
    {
        mu = weights * eta;
    }

    void hessian(
        const Eigen::Ref<const vec_value_t>&,
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> var
    ) override
    {
        var = weights;
    }

    value_t deviance(
        const Eigen::Ref<const vec_value_t>& y,
        const Eigen::Ref<const vec_value_t>& eta,
        const Eigen::Ref<const vec_value_t>& weights
    ) override
    {
        return (weights * (0.5 * eta.square() - y * eta)).sum();
    }

    value_t deviance_full(
        const Eigen::Ref<const vec_value_t>& y,
        const Eigen::Ref<const vec_value_t>& weights
    ) override
    {
        return -0.5 * (y.square() * weights).sum();
    }
};

} // namespace glm
} // namespace adelie_core