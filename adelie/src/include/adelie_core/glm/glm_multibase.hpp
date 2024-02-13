#pragma once
#include <string>
#include <adelie_core/util/types.hpp>

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

    const string_t name;
    const bool is_multi = true;
    const bool is_symmetric = false;

    explicit GlmMultiBase(
        const string_t& name,
        bool is_symmetric
    ):
        name(name),
        is_symmetric(is_symmetric)
    {}

    virtual ~GlmMultiBase() =default;

    virtual void gradient(
        const Eigen::Ref<const rowarr_value_t>& eta,
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<rowarr_value_t> mu
    ) =0;

    virtual void hessian(
        const Eigen::Ref<const rowarr_value_t>& mu,
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<rowarr_value_t> var
    ) =0;

    virtual value_t loss(
        const Eigen::Ref<const rowarr_value_t>& y,
        const Eigen::Ref<const rowarr_value_t>& eta,
        const Eigen::Ref<const vec_value_t>& weights
    ) =0;

    virtual value_t loss_full(
        const Eigen::Ref<const rowarr_value_t>& y,
        const Eigen::Ref<const vec_value_t>& weights
    ) =0;
};

} // namespace glm
} // namespace adelie_core