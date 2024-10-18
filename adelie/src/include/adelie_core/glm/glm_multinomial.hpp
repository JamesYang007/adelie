#pragma once
#include <adelie_core/glm/glm_multibase.hpp>

#ifndef ADELIE_CORE_GLM_MULTINOMIAL_TP
#define ADELIE_CORE_GLM_MULTINOMIAL_TP \
    template <class ValueType>
#endif
#ifndef ADELIE_CORE_GLM_MULTINOMIAL
#define ADELIE_CORE_GLM_MULTINOMIAL \
    GlmMultinomial<ValueType>
#endif

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
    );

    ADELIE_CORE_GLM_MULTI_PURE_OVERRIDE_DECL
};

} // namespace glm
} // namespace adelie_core