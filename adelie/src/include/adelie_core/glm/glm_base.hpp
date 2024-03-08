#pragma once
#include <cstdio>
#include <string>
#include <adelie_core/configs.hpp>
#include <adelie_core/util/types.hpp>
#include <adelie_core/util/format.hpp>

namespace adelie_core {
namespace glm {

template <class ValueType>
class GlmBase
{
public:
    using value_t = ValueType;
    using string_t = std::string;
    using vec_value_t = util::rowvec_type<value_t>;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;

    const string_t name;
    map_cvec_value_t y;
    map_cvec_value_t weights;
    const bool is_multi = false;
    const bool is_symmetric = false;

protected:
    void check_gradient(
        const Eigen::Ref<const vec_value_t>& eta,
        const Eigen::Ref<const vec_value_t>& grad
    ) const
    {
        if (
            (weights.size() != y.size()) ||
            (weights.size() != eta.size()) ||
            (weights.size() != grad.size())
        ) {
            throw std::runtime_error(
                util::format(
                    "gradient() is given inconsistent inputs! "
                    "(weights=%d, y=%d, eta=%d, grad=%d)",
                    weights.size(), y.size(), eta.size(), grad.size()
                )
            );
        }
    }

    void check_hessian(
        const Eigen::Ref<const vec_value_t>& eta,
        const Eigen::Ref<const vec_value_t>& grad,
        const Eigen::Ref<const vec_value_t>& hess
    ) const
    {
        if (
            (weights.size() != y.size()) ||
            (weights.size() != eta.size()) ||
            (weights.size() != grad.size()) ||
            (weights.size() != hess.size())
        ) {
            throw std::runtime_error(
                util::format(
                    "hessian() is given inconsistent inputs! "
                    "(weights=%d, y=%d, eta=%d, grad=%d, hess=%d)",
                    weights.size(), y.size(), eta.size(), grad.size(), hess.size()
                )
            );
        }
    }

    void check_inv_hessian_gradient(
        const Eigen::Ref<const vec_value_t>& eta,
        const Eigen::Ref<const vec_value_t>& grad,
        const Eigen::Ref<const vec_value_t>& hess,
        const Eigen::Ref<const vec_value_t>& inv_hess_grad
    ) const
    {
        if (
            (weights.size() != y.size()) ||
            (weights.size() != eta.size()) ||
            (weights.size() != grad.size()) ||
            (weights.size() != hess.size()) ||
            (weights.size() != inv_hess_grad.size())
        ) {
            throw std::runtime_error(
                util::format(
                    "inv_hessian_grad() is given inconsistent inputs! "
                    "(weights=%d, y=%d, eta=%d, grad=%d, hess=%d, inv_hess_grad=%d)",
                    weights.size(), y.size(), eta.size(), grad.size(), hess.size(), inv_hess_grad.size()
                )
            );
        }
    }

    void check_loss(
        const Eigen::Ref<const vec_value_t>& eta
    ) const
    {
        if (
            (y.size() != weights.size()) ||
            (y.size() != eta.size())
        ) {
            throw std::runtime_error(
                util::format(
                    "loss() is given inconsistent inputs! "
                    "(y=%d, weights=%d, eta=%d)",
                    y.size(), weights.size(), eta.size()
                )
            );
        }
    }

public:
    explicit GlmBase(
        const string_t& name,
        const Eigen::Ref<const vec_value_t>& y,
        const Eigen::Ref<const vec_value_t>& weights
    ):
        name(name),
        y(y.data(), y.size()),
        weights(weights.data(), weights.size())
    {
        if (y.size() != weights.size()) {
            throw std::runtime_error("y and weights must have same length.");
        }
    }

    virtual ~GlmBase() =default;

    virtual void gradient(
        const Eigen::Ref<const vec_value_t>& eta,
        Eigen::Ref<vec_value_t> grad
    ) =0;

    virtual void hessian(
        const Eigen::Ref<const vec_value_t>& eta,
        const Eigen::Ref<const vec_value_t>& grad,
        Eigen::Ref<vec_value_t> hess
    ) =0;

    virtual void inv_hessian_gradient(
        const Eigen::Ref<const vec_value_t>& eta,
        const Eigen::Ref<const vec_value_t>& grad,
        const Eigen::Ref<const vec_value_t>& hess,
        Eigen::Ref<vec_value_t> inv_hess_grad
    )
    {
        check_inv_hessian_gradient(eta, grad, hess, inv_hess_grad);
        inv_hess_grad = grad / (
            hess.max(0) + 
            value_t(Configs::hessian_min) * (hess <= 0).template cast<value_t>()
        );
    }

    virtual value_t loss(
        const Eigen::Ref<const vec_value_t>& eta
    ) =0;

    virtual value_t loss_full() =0;
};

} // namespace glm
} // namespace adelie_core