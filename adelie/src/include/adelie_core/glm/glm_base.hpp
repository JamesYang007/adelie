#pragma once
#include <string>
#include <adelie_core/util/exceptions.hpp>
#include <adelie_core/util/format.hpp>
#include <adelie_core/util/types.hpp>

#ifndef ADELIE_CORE_GLM_BASE_TP
#define ADELIE_CORE_GLM_BASE_TP \
    template <class ValueType>
#endif
#ifndef ADELIE_CORE_GLM_BASE
#define ADELIE_CORE_GLM_BASE \
    GlmBase<ValueType>
#endif

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

protected:
    inline void check_gradient(
        const Eigen::Ref<const vec_value_t>& eta,
        const Eigen::Ref<const vec_value_t>& grad
    ) const;

    inline void check_hessian(
        const Eigen::Ref<const vec_value_t>& eta,
        const Eigen::Ref<const vec_value_t>& grad,
        const Eigen::Ref<const vec_value_t>& hess
    ) const;

    inline void check_inv_hessian_gradient(
        const Eigen::Ref<const vec_value_t>& eta,
        const Eigen::Ref<const vec_value_t>& grad,
        const Eigen::Ref<const vec_value_t>& hess,
        const Eigen::Ref<const vec_value_t>& inv_hess_grad
    ) const;

    inline void check_loss(
        const Eigen::Ref<const vec_value_t>& eta
    ) const;

public:
    explicit GlmBase(
        const string_t& name,
        const Eigen::Ref<const vec_value_t>& y,
        const Eigen::Ref<const vec_value_t>& weights
    );

    virtual ~GlmBase() {};

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
    );

    virtual value_t loss(
        const Eigen::Ref<const vec_value_t>& eta
    ) =0;

    virtual value_t loss_full() =0;

    virtual void inv_link(
        const Eigen::Ref<const vec_value_t>& eta,
        Eigen::Ref<vec_value_t> out
    ) =0;
};

ADELIE_CORE_GLM_BASE_TP
void 
ADELIE_CORE_GLM_BASE::check_gradient(
    const Eigen::Ref<const vec_value_t>& eta,
    const Eigen::Ref<const vec_value_t>& grad
) const
{
    if (
        (weights.size() != y.size()) ||
        (weights.size() != eta.size()) ||
        (weights.size() != grad.size())
    ) {
        throw util::adelie_core_error(
            util::format(
                "gradient() is given inconsistent inputs! "
                "(weights=%d, y=%d, eta=%d, grad=%d)",
                weights.size(), y.size(), eta.size(), grad.size()
            )
        );
    }
}

ADELIE_CORE_GLM_BASE_TP
void 
ADELIE_CORE_GLM_BASE::check_hessian(
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
        throw util::adelie_core_error(
            util::format(
                "hessian() is given inconsistent inputs! "
                "(weights=%d, y=%d, eta=%d, grad=%d, hess=%d)",
                weights.size(), y.size(), eta.size(), grad.size(), hess.size()
            )
        );
    }
}

ADELIE_CORE_GLM_BASE_TP
void 
ADELIE_CORE_GLM_BASE::check_inv_hessian_gradient(
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
        throw util::adelie_core_error(
            util::format(
                "inv_hessian_grad() is given inconsistent inputs! "
                "(weights=%d, y=%d, eta=%d, grad=%d, hess=%d, inv_hess_grad=%d)",
                weights.size(), y.size(), eta.size(), grad.size(), hess.size(), inv_hess_grad.size()
            )
        );
    }
}

ADELIE_CORE_GLM_BASE_TP
void 
ADELIE_CORE_GLM_BASE::check_loss(
    const Eigen::Ref<const vec_value_t>& eta
) const
{
    if (
        (y.size() != weights.size()) ||
        (y.size() != eta.size())
    ) {
        throw util::adelie_core_error(
            util::format(
                "loss() is given inconsistent inputs! "
                "(y=%d, weights=%d, eta=%d)",
                y.size(), weights.size(), eta.size()
            )
        );
    }
}

} // namespace glm
} // namespace adelie_core

#ifndef ADELIE_CORE_GLM_PURE_OVERRIDE_DECL
#define ADELIE_CORE_GLM_PURE_OVERRIDE_DECL \
    void gradient(\
        const Eigen::Ref<const vec_value_t>& eta,\
        Eigen::Ref<vec_value_t> grad\
    ) override;\
    void hessian(\
        const Eigen::Ref<const vec_value_t>& eta,\
        const Eigen::Ref<const vec_value_t>& grad,\
        Eigen::Ref<vec_value_t> hess\
    ) override;\
    value_t loss(\
        const Eigen::Ref<const vec_value_t>& eta\
    ) override;\
    value_t loss_full() override;\
    void inv_link(\
        const Eigen::Ref<const vec_value_t>& eta,\
        Eigen::Ref<vec_value_t> out\
    ) override;
#endif