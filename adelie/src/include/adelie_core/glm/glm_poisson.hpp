#pragma once
#include <adelie_core/glm/glm_base.hpp>

namespace adelie_core {
namespace glm {

template <class ValueType>
class GlmPoisson: public GlmBase<ValueType>
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
        mu = eta.exp();
    }

    void gradient_inverse(
        const Eigen::Ref<const vec_value_t>& mu,
        Eigen::Ref<vec_value_t> eta
    ) override
    {
        eta = mu.log();
    }

    void hessian(
        const Eigen::Ref<const vec_value_t>& mu,
        Eigen::Ref<vec_value_t> var
    ) override
    {
        var = mu;
    }

    void deviance(
        const Eigen::Ref<const vec_value_t>& y,
        const Eigen::Ref<const vec_value_t>& eta,
        Eigen::Ref<vec_value_t> dev
    ) override
    {
        // numerically stable when y == 0 and eta could be -inf
        dev = (-eta).min(std::numeric_limits<value_t>::max()) * y + eta.exp();
    }
};

} // namespace glm
} // namespace adelie_core