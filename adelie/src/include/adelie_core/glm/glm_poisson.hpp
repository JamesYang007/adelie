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

    explicit GlmPoisson():
        base_t("poisson")
    {}

    void gradient(
        const Eigen::Ref<const vec_value_t>& eta,
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> mu
    ) override
    {
        mu = weights * eta.exp();
    }

    void hessian(
        const Eigen::Ref<const vec_value_t>& mu,
        const Eigen::Ref<const vec_value_t>&,
        Eigen::Ref<vec_value_t> var
    ) override
    {
        var = mu;
    }

    value_t deviance(
        const Eigen::Ref<const vec_value_t>& y,
        const Eigen::Ref<const vec_value_t>& eta,
        const Eigen::Ref<const vec_value_t>& weights
    ) override
    {
        // numerically stable when y == 0 and eta could be -inf
        return (weights * ((-eta).min(std::numeric_limits<value_t>::max()) * y + eta.exp())).sum();
    }

    value_t deviance_full(
        const Eigen::Ref<const vec_value_t>& y,
        const Eigen::Ref<const vec_value_t>& weights
    ) override
    {
        return (weights * ((-y.log()).min(std::numeric_limits<value_t>::max()) * y + y)).sum();
    }
};

} // namespace glm
} // namespace adelie_core