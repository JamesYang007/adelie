#pragma once
#include <adelie_core/glm/glm_base.hpp>

#ifndef ADELIE_CORE_GLM_POISSON_TP
#define ADELIE_CORE_GLM_POISSON_TP \
    template <class ValueType>
#endif
#ifndef ADELIE_CORE_GLM_POISSON
#define ADELIE_CORE_GLM_POISSON \
    GlmPoisson<ValueType>
#endif

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
    );

    ADELIE_CORE_GLM_PURE_OVERRIDE_DECL
};

} // namespace glm
} // namespace adelie_core