#pragma once
#include <string>
#include <adelie_core/util/exceptions.hpp>
#include <adelie_core/util/format.hpp>
#include <adelie_core/util/types.hpp>

#ifndef ADELIE_CORE_GLM_MULTIBASE_TP
#define ADELIE_CORE_GLM_MULTIBASE_TP \
    template <class ValueType>
#endif
#ifndef ADELIE_CORE_GLM_MULTIBASE
#define ADELIE_CORE_GLM_MULTIBASE \
    GlmMultiBase<ValueType>
#endif

namespace adelie_core {
namespace glm {

template <class ValueType>
class GlmMultiBase
{
public:
    using value_t = ValueType;
    using string_t = std::string;
    using vec_value_t = util::rowvec_type<value_t>;
    using rowarr_value_t = util::rowarr_type<value_t>;
    using map_carr_value_t = Eigen::Map<const rowarr_value_t>;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;

    const string_t name;
    map_carr_value_t y;
    map_cvec_value_t weights;
    const bool is_multi = true;

protected:
    inline void check_gradient(
        const Eigen::Ref<const rowarr_value_t>& eta,
        const Eigen::Ref<const rowarr_value_t>& grad
    ) const;

    inline void check_hessian(
        const Eigen::Ref<const rowarr_value_t>& eta,
        const Eigen::Ref<const rowarr_value_t>& grad,
        const Eigen::Ref<const rowarr_value_t>& hess
    ) const;

    inline void check_inv_hessian_gradient(
        const Eigen::Ref<const rowarr_value_t>& eta,
        const Eigen::Ref<const rowarr_value_t>& grad,
        const Eigen::Ref<const rowarr_value_t>& hess,
        const Eigen::Ref<const rowarr_value_t>& inv_hess_grad
    ) const;

    inline void check_loss(
        const Eigen::Ref<const rowarr_value_t>& eta
    ) const;

public:
    explicit GlmMultiBase(
        const string_t& name,
        const Eigen::Ref<const rowarr_value_t>& y,
        const Eigen::Ref<const vec_value_t>& weights
    );

    virtual ~GlmMultiBase() {};

    virtual void gradient(
        const Eigen::Ref<const rowarr_value_t>& eta,
        Eigen::Ref<rowarr_value_t> grad
    ) =0;

    virtual void hessian(
        const Eigen::Ref<const rowarr_value_t>& eta,
        const Eigen::Ref<const rowarr_value_t>& grad,
        Eigen::Ref<rowarr_value_t> hess
    ) =0;

    virtual void inv_hessian_gradient(
        const Eigen::Ref<const rowarr_value_t>& eta,
        const Eigen::Ref<const rowarr_value_t>& grad,
        const Eigen::Ref<const rowarr_value_t>& hess,
        Eigen::Ref<rowarr_value_t> inv_hess_grad
    );

    virtual value_t loss(
        const Eigen::Ref<const rowarr_value_t>& eta
    ) =0;

    virtual value_t loss_full() =0;

    virtual void inv_link(
        const Eigen::Ref<const rowarr_value_t>& eta,
        Eigen::Ref<rowarr_value_t> out
    ) =0;
};

ADELIE_CORE_GLM_MULTIBASE_TP
void
ADELIE_CORE_GLM_MULTIBASE::check_gradient(
    const Eigen::Ref<const rowarr_value_t>& eta,
    const Eigen::Ref<const rowarr_value_t>& grad
) const
{
    if (
        (weights.size() != y.rows()) ||
        (weights.size() != eta.rows()) ||
        (weights.size() != grad.rows()) ||
        (eta.cols() != y.cols()) ||
        (eta.cols() != grad.cols())
    ) {
        throw util::adelie_core_error(
            util::format(
                "gradient() is given inconsistent inputs! "
                "(weights=%d, y=(%d, %d), eta=(%d, %d), grad=(%d, %d))",
                weights.size(), y.rows(), y.cols(), eta.rows(), eta.cols(), grad.rows(), grad.cols()
            )
        );
    }
}

ADELIE_CORE_GLM_MULTIBASE_TP
void
ADELIE_CORE_GLM_MULTIBASE::check_hessian(
    const Eigen::Ref<const rowarr_value_t>& eta,
    const Eigen::Ref<const rowarr_value_t>& grad,
    const Eigen::Ref<const rowarr_value_t>& hess
) const
{
    if (
        (weights.size() != y.rows()) ||
        (weights.size() != eta.rows()) ||
        (weights.size() != grad.rows()) ||
        (weights.size() != hess.rows()) ||
        (eta.cols() != y.cols()) ||
        (eta.cols() != grad.cols()) ||
        (eta.cols() != hess.cols())
    ) {
        throw util::adelie_core_error(
            util::format(
                "hessian() is given inconsistent inputs! "
                "(weights=%d, y=(%d, %d), eta=(%d, %d), grad=(%d, %d), hess=(%d, %d))",
                weights.size(), y.rows(), y.cols(), eta.rows(), eta.cols(), 
                grad.rows(), grad.cols(), hess.rows(), hess.cols()
            )
        );
    }
}

ADELIE_CORE_GLM_MULTIBASE_TP
void
ADELIE_CORE_GLM_MULTIBASE::check_inv_hessian_gradient(
    const Eigen::Ref<const rowarr_value_t>& eta,
    const Eigen::Ref<const rowarr_value_t>& grad,
    const Eigen::Ref<const rowarr_value_t>& hess,
    const Eigen::Ref<const rowarr_value_t>& inv_hess_grad
) const
{
    if (
        (weights.size() != y.rows()) ||
        (weights.size() != eta.rows()) ||
        (weights.size() != grad.rows()) ||
        (weights.size() != hess.rows()) ||
        (weights.size() != inv_hess_grad.rows()) ||
        (eta.cols() != y.cols()) ||
        (eta.cols() != grad.cols()) ||
        (eta.cols() != hess.cols()) ||
        (eta.cols() != inv_hess_grad.cols())
    ) {
        throw util::adelie_core_error(
            util::format(
                "inv_hessian_gradient() is given inconsistent inputs! "
                "(weights=%d, y=(%d, %d), eta=(%d, %d), grad=(%d, %d), hess=(%d, %d), inv_hess_grad=(%d, %d))",
                weights.size(), y.rows(), y.cols(), eta.rows(), eta.cols(), 
                grad.rows(), grad.cols(), hess.rows(), hess.cols(),
                inv_hess_grad.rows(), inv_hess_grad.cols()
            )
        );
    }
}

ADELIE_CORE_GLM_MULTIBASE_TP
void
ADELIE_CORE_GLM_MULTIBASE::check_loss(
    const Eigen::Ref<const rowarr_value_t>& eta
) const
{
    if (
        (y.rows() != weights.size()) ||
        (y.rows() != eta.rows()) ||
        (y.cols() != eta.cols())
    ) {
        throw util::adelie_core_error(
            util::format(
                "loss() is given inconsistent inputs! "
                "(y=(%d, %d), weights=%d, eta=(%d, %d))",
                y.rows(), y.cols(), weights.size(), eta.rows(), eta.cols()
            )
        );
    }
}

} // namespace glm
} // namespace adelie_core

#ifndef ADELIE_CORE_GLM_MULTI_PURE_OVERRIDE_DECL
#define ADELIE_CORE_GLM_MULTI_PURE_OVERRIDE_DECL\
    void gradient(\
        const Eigen::Ref<const rowarr_value_t>& eta,\
        Eigen::Ref<rowarr_value_t> grad\
    ) override;\
    void hessian(\
        const Eigen::Ref<const rowarr_value_t>& eta,\
        const Eigen::Ref<const rowarr_value_t>& grad,\
        Eigen::Ref<rowarr_value_t> hess\
    ) override;\
    value_t loss(\
        const Eigen::Ref<const rowarr_value_t>& eta\
    ) override;\
    value_t loss_full() override;\
    void inv_link(\
        const Eigen::Ref<const rowarr_value_t>& eta,\
        Eigen::Ref<rowarr_value_t> out\
    ) override;
#endif