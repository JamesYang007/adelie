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
    using base_t::y;
    using base_t::weights;

    explicit GlmPoisson(
        const Eigen::Ref<const vec_value_t>& y,
        const Eigen::Ref<const vec_value_t>& weights
    ):
        base_t("poisson", y, weights)
    {}

    void gradient(
        const Eigen::Ref<const vec_value_t>& eta,
        Eigen::Ref<vec_value_t> grad
    ) override
    {
        base_t::check_gradient(eta, grad);
        grad = weights * (y - eta.exp());
    }

    void hessian(
        const Eigen::Ref<const vec_value_t>& eta,
        const Eigen::Ref<const vec_value_t>& grad,
        Eigen::Ref<vec_value_t> hess
    ) override
    {
        base_t::check_hessian(eta, grad, hess);
        hess = weights * y - grad;
    }

    value_t loss(
        const Eigen::Ref<const vec_value_t>& eta
    ) override
    {
        base_t::check_loss(eta);
        // numerically stable when y == 0 and eta could be -inf
        return (weights * ((-eta).min(std::numeric_limits<value_t>::max()) * y + eta.exp())).sum();
    }

    value_t loss_full() override
    {
        return (weights * ((-y.log()).min(std::numeric_limits<value_t>::max()) * y + y)).sum();
    }
};

} // namespace glm
} // namespace adelie_core