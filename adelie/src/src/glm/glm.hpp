#pragma once
#include <adelie_core/glm/glm_base.hpp>
#include <adelie_core/glm/glm_binomial.hpp>
#include <adelie_core/glm/glm_cox.hpp>
#include <adelie_core/glm/glm_gaussian.hpp>
#include <adelie_core/glm/glm_multibase.hpp>
#include <adelie_core/glm/glm_multigaussian.hpp>
#include <adelie_core/glm/glm_multinomial.hpp>
#include <adelie_core/glm/glm_poisson.hpp>

extern template class adelie_core::glm::GlmBase<float>;
extern template class adelie_core::glm::GlmBase<double>;

extern template class adelie_core::glm::GlmBinomialLogit<float>;
extern template class adelie_core::glm::GlmBinomialLogit<double>;
extern template class adelie_core::glm::GlmBinomialProbit<float>;
extern template class adelie_core::glm::GlmBinomialProbit<double>;

extern template class adelie_core::glm::GlmCox<float>;
extern template class adelie_core::glm::GlmCox<double>;

extern template class adelie_core::glm::GlmGaussian<float>;
extern template class adelie_core::glm::GlmGaussian<double>;

extern template class adelie_core::glm::GlmMultiBase<float>;
extern template class adelie_core::glm::GlmMultiBase<double>;

extern template class adelie_core::glm::GlmMultiGaussian<float>;
extern template class adelie_core::glm::GlmMultiGaussian<double>;

extern template class adelie_core::glm::GlmMultinomial<float>;
extern template class adelie_core::glm::GlmMultinomial<double>;

extern template class adelie_core::glm::GlmPoisson<float>;
extern template class adelie_core::glm::GlmPoisson<double>;