#pragma once
#include <adelie_core/glm/glm_base.hpp>

#ifndef ADELIE_CORE_GLM_BINOMIAL_LOGIT_TP
#define ADELIE_CORE_GLM_BINOMIAL_LOGIT_TP \
    template <class ValueType>
#endif
#ifndef ADELIE_CORE_GLM_BINOMIAL_LOGIT
#define ADELIE_CORE_GLM_BINOMIAL_LOGIT \
    GlmBinomialLogit<ValueType>
#endif

#ifndef ADELIE_CORE_GLM_BINOMIAL_PROBIT_TP
#define ADELIE_CORE_GLM_BINOMIAL_PROBIT_TP \
    template <class ValueType>
#endif
#ifndef ADELIE_CORE_GLM_BINOMIAL_PROBIT
#define ADELIE_CORE_GLM_BINOMIAL_PROBIT \
    GlmBinomialProbit<ValueType>
#endif

namespace adelie_core {
namespace glm {

template <class ValueType>
class GlmBinomialLogit: public GlmBase<ValueType>
{
public:
    using base_t = GlmBase<ValueType>;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using base_t::y;
    using base_t::weights;

    explicit GlmBinomialLogit(
        const Eigen::Ref<const vec_value_t>& y,
        const Eigen::Ref<const vec_value_t>& weights
    );

    ADELIE_CORE_GLM_PURE_OVERRIDE_DECL
};

template <class ValueType>
class GlmBinomialProbit: public GlmBase<ValueType>
{
public:
    using base_t = GlmBase<ValueType>;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using base_t::y;
    using base_t::weights;

private:
    static inline auto std_cdf(
        const Eigen::Ref<const vec_value_t>& x
    );

    static inline auto std_pdf(
        const Eigen::Ref<const vec_value_t>& x
    );

    vec_value_t _buff;

public:
    explicit GlmBinomialProbit(
        const Eigen::Ref<const vec_value_t>& y,
        const Eigen::Ref<const vec_value_t>& weights
    );
    
    ADELIE_CORE_GLM_PURE_OVERRIDE_DECL
};

} // namespace glm
} // namespace adelie_core