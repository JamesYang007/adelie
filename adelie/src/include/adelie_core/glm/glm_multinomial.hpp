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

public:
    explicit GlmMultinomial(
        const Eigen::Ref<const rowarr_value_t>& y,
        const Eigen::Ref<const vec_value_t>& weights
    ):
        base_t("multinomial", y, weights),
        _buff(y.rows() * (y.cols() + 1))
    {
        if (y.cols() <= 1) {
            throw util::adelie_core_error(
                "y must have at least 2 columns (classes)."
            );
        }
    }

    void gradient(
        const Eigen::Ref<const rowarr_value_t>& eta,
        Eigen::Ref<rowarr_value_t> grad
    ) override
    {
        base_t::check_gradient(eta, grad);
        Eigen::Map<vec_value_t> eta_max(_buff.data(), y.rows());
        eta_max = eta.rowwise().maxCoeff();
        grad = (eta.colwise() - eta_max.matrix().transpose().array()).exp();
        auto& sum_exp = eta_max;
        sum_exp = grad.rowwise().sum();
        grad = (
            (y - grad.colwise() / sum_exp.matrix().transpose().array()).colwise() * 
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
        // 2 * K^{-1} W[:, None] * P * (1 - P)
        hess *= 2 * (1 - grad.cols() * (
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
        Eigen::Map<vec_value_t> eta_max(_buff.data(), y.rows());
        eta_max = eta.rowwise().maxCoeff();
        Eigen::Map<rowarr_value_t> eta_shift(_buff.data() + y.rows(), y.rows(), y.cols());
        eta_shift = (eta.colwise() - eta_max.matrix().transpose().array());
        return (
            weights.matrix().transpose().array() * (
                - (y * eta_shift).rowwise().sum()
                + eta_shift.exp().rowwise().sum().log()
            )
        ).sum() / y.cols();
    }

    value_t loss_full() override
    {
        value_t loss = 0;
        for (int i = 0; i < y.rows(); ++i) {
            value_t sum = 0;
            for (int k = 0; k < y.cols(); ++k) {
                const auto log_yik = std::log(y(i,k));
                if (!(std::isinf(log_yik) || std::isnan(log_yik))) {
                    sum += y(i,k) * log_yik;
                }
            }
            loss -= sum * weights[i] / y.cols();
        }    
        return loss;
    }
};

} // namespace glm
} // namespace adelie_core