#pragma once
#include <adelie_core/glm/glm_multibase.hpp>

#ifndef ADELIE_CORE_GLM_MULTIGAUSSIAN_TP
#define ADELIE_CORE_GLM_MULTIGAUSSIAN_TP \
    template <class ValueType>
#endif
#ifndef ADELIE_CORE_GLM_MULTIGAUSSIAN
#define ADELIE_CORE_GLM_MULTIGAUSSIAN \
    GlmMultiGaussian<ValueType>
#endif

namespace adelie_core {
namespace glm {

template <class ValueType>
class GlmMultiGaussian: public GlmMultiBase<ValueType>
{
public:
    using base_t = GlmMultiBase<ValueType>;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using typename base_t::rowarr_value_t;
    using base_t::y;
    using base_t::weights;

    explicit GlmMultiGaussian(
        const Eigen::Ref<const rowarr_value_t>& y,
        const Eigen::Ref<const vec_value_t>& weights
    );

    ADELIE_CORE_GLM_MULTI_PURE_OVERRIDE_DECL
};

} // namespace glm
} // namespace adelie_core