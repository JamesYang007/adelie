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

    explicit GlmBinomial():
        base_t("binomial")
    {}

    void gradient(
        const Eigen::Ref<const vec_value_t>& eta,
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> mu
    ) override
    {
        mu = weights / (1 + (-eta).exp());
    }

    void hessian(
        const Eigen::Ref<const vec_value_t>& mu,
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> var
    ) override
    {
        // denominator is given a dummy positive value when weights == 0.
        var = (mu * (weights-mu)) / (weights + (weights <= 0).template cast<value_t>());
    }

    value_t loss(
        const Eigen::Ref<const vec_value_t>& y,
        const Eigen::Ref<const vec_value_t>& eta,
        const Eigen::Ref<const vec_value_t>& weights
    ) override
    {
        // numerically stable so that exp == 0 when y = 1 and eta = inf or y = 0 and eta = -inf.
        // NOTE: this only works when y takes values in {0,1}
        return (weights * (1 + ((1-2*y) * eta).exp()).log()).sum();
    }

    value_t loss_full(
        const Eigen::Ref<const vec_value_t>&,
        const Eigen::Ref<const vec_value_t>& 
    ) override
    {
        return 0;
    }
};

} // namespace glm
} // namespace adelie_core