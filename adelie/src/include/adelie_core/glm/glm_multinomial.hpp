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
    using base_t::y;
    using base_t::weights;

private:
    vec_value_t _buff;
    rowarr_value_t _buff2;

public:
    explicit GlmMultinomial(
        const Eigen::Ref<const rowarr_value_t>& y,
        const Eigen::Ref<const vec_value_t>& weights
    ):
        base_t("multinomial", y, weights),
        _buff(weights.size()),
        _buff2(y.rows(), y.cols())
    {}

    void gradient(
        const Eigen::Ref<const rowarr_value_t>& eta,
        Eigen::Ref<rowarr_value_t> grad
    ) override
    {
        base_t::check_gradient(eta, grad);
        _buff = eta.rowwise().maxCoeff();
        grad = (eta.colwise() - _buff.matrix().transpose().array()).exp();
        _buff = grad.rowwise().sum();
        grad = (
            (y - grad.colwise() / _buff.matrix().transpose().array()).colwise() * 
            weights.matrix().transpose().array() / eta.cols()
        );
    }

    void hessian(
        const Eigen::Ref<const rowarr_value_t>& eta,
        const Eigen::Ref<const rowarr_value_t>& grad,
        Eigen::Ref<rowarr_value_t> hess
    ) override
    {
        base_t::check_hessian(eta, grad, hess);
         // K^{-1} W[:, None] * P
        hess = (
            y.colwise() * weights.matrix().transpose().array() / eta.cols() 
            - grad
        );
        // K^{-1} W[:, None] * P * (1 - P)
        hess = hess * (1 - grad.cols() * (
                hess.colwise() /
                (weights + (weights <= 0).template cast<value_t>()).matrix().transpose().array()
            )
        );
    }

    value_t loss(
        const Eigen::Ref<const rowarr_value_t>& eta
    ) override
    {
        base_t::check_loss(eta);
        _buff = eta.rowwise().maxCoeff();
        _buff2 = (eta.colwise() - _buff.matrix().transpose().array());
        return (
            weights.matrix().transpose().array() * (
                - (y * _buff2).rowwise().sum()
                + _buff2.exp().rowwise().sum().log()
            )
        ).sum() / y.cols();
    }

    value_t loss_full() override
    {
        return 0;
    }
};

} // namespace glm
} // namespace adelie_core