#pragma once
#include <adelie_core/glm/glm_multibase.hpp>

namespace adelie_core {
namespace glm {

template <class ValueType>
class GlmMultinomial: public GlmMultiBase<ValueType>
{
public:
    using base_t = GlmMultiBase<ValueType>;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using typename base_t::rowarr_value_t;

private:
    vec_value_t _buff;

public:
    explicit GlmMultinomial():
        base_t("multinomial", true)
    {}

    void gradient(
        const Eigen::Ref<const rowarr_value_t>& eta,
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<rowarr_value_t> mu
    ) override
    {
        mu = (eta.colwise() - eta.rowwise().maxCoeff()).exp();
        _buff = weights.matrix().transpose().array() / (eta.cols() * mu.rowwise().sum());
        mu.colwise() *= _buff.matrix().transpose().array();
    }

    void hessian(
        const Eigen::Ref<const rowarr_value_t>& mu,
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<rowarr_value_t> var
    ) override
    {
        var = mu.cols() * (
            mu.colwise() /
            (weights + (weights <= 0).template cast<value_t>()).matrix().transpose().array()
        );
        var = mu * (1 - var);
    }

    value_t loss(
        const Eigen::Ref<const rowarr_value_t>& y,
        const Eigen::Ref<const rowarr_value_t>& eta,
        const Eigen::Ref<const vec_value_t>& weights
    ) override
    {
        _buff = eta.rowwise().maxCoeff();
        // TODO: this gets evaluated twice below.
        // For simplicity, we keep this code since this is not speed critical.
        const auto eta_relative = (eta.colwise() - _buff.matrix().transpose().array());
        return (
            weights.matrix().transpose().array() * (
                - (y * eta_relative).rowwise().sum()
                + eta_relative.exp().rowwise().sum().log()
            )
        ).sum() / y.cols();
    }

    value_t loss_full(
        const Eigen::Ref<const rowarr_value_t>&,
        const Eigen::Ref<const vec_value_t>& 
    ) override
    {
        return 0;
    }
};

} // namespace glm
} // namespace adelie_core