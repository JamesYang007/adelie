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

    void gradient(
        const Eigen::Ref<const vec_value_t>& eta,
        Eigen::Ref<vec_value_t> mu
    ) override
    {
        mu = 1 / (1 + (-eta).exp());
    }

    void gradient_inverse(
        const Eigen::Ref<const vec_value_t>& mu,
        Eigen::Ref<vec_value_t> eta
    ) override
    {
        eta = (mu / (1-mu)).log();
    }

    void hessian(
        const Eigen::Ref<const vec_value_t>& eta,
        Eigen::Ref<vec_value_t> var
    ) override
    {
        var = 1 / (1 + (-eta).exp());
        var *= (1-var);
    }

    void deviance(
        const Eigen::Ref<const vec_value_t>& y,
        const Eigen::Ref<const vec_value_t>& eta,
        Eigen::Ref<vec_value_t> dev
    ) override
    {
        dev = (1 + ((1-2*y) * eta).exp()).log();
    }
};

} // namespace glm
} // namespace adelie_core