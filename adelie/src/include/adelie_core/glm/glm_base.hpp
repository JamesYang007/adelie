#pragma once
#include <adelie_core/util/types.hpp>

namespace adelie_core {
namespace glm {

template <class ValueType>
class GlmBase
{
public:
    using value_t = ValueType;
    using vec_value_t = util::rowvec_type<value_t>;

    virtual ~GlmBase() =default;

    virtual void gradient(
        const Eigen::Ref<const vec_value_t>& eta,
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> mu
    ) =0;

    virtual void hessian(
        const Eigen::Ref<const vec_value_t>& mu,
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> var
    ) =0;

    virtual value_t deviance(
        const Eigen::Ref<const vec_value_t>& y,
        const Eigen::Ref<const vec_value_t>& eta,
        const Eigen::Ref<const vec_value_t>& weights
    ) =0;

    virtual value_t deviance_full(
        const Eigen::Ref<const vec_value_t>& y,
        const Eigen::Ref<const vec_value_t>& weights
    ) =0;
};

} // namespace glm
} // namespace adelie_core