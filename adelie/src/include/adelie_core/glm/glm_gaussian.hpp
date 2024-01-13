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

    void gradient(
        const Eigen::Ref<const vec_value_t>& eta,
        Eigen::Ref<vec_value_t> mu
    ) override
    {
        mu = eta;
    }

    void gradient_inverse(
        const Eigen::Ref<const vec_value_t>& mu,
        Eigen::Ref<vec_value_t> eta
    ) override
    {
        eta = mu;
    }

    void hessian(
        const Eigen::Ref<const vec_value_t>&,
        Eigen::Ref<vec_value_t> var
    ) override
    {
        var = 1;
    }

    void deviance(
        const Eigen::Ref<const vec_value_t>& y,
        const Eigen::Ref<const vec_value_t>& eta,
        Eigen::Ref<vec_value_t> dev
    ) override
    {
        dev = -y * eta + 0.5 * eta.square();
    }
};

} // namespace glm
} // namespace adelie_core