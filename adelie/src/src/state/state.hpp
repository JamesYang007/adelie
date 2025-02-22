#pragma once
#include <tools/types.hpp>
#include <adelie_core/state/state_bvls.hpp>
#include <adelie_core/state/state_css_cov.hpp>
#include <adelie_core/state/state_gaussian_cov.hpp>
#include <adelie_core/state/state_gaussian_naive.hpp>
#include <adelie_core/state/state_gaussian_pin_cov.hpp>
#include <adelie_core/state/state_gaussian_pin_naive.hpp>
#include <adelie_core/state/state_glm_naive.hpp>
#include <adelie_core/state/state_multigaussian_naive.hpp>
#include <adelie_core/state/state_multiglm_naive.hpp>
#include <adelie_core/state/state_pinball.hpp>

extern template class adelie_core::state::StateBase<constraint_type<float>>;
extern template class adelie_core::state::StateBase<constraint_type<double>>;

extern template class adelie_core::state::StateBVLS<matrix_naive_type<float>>;
extern template class adelie_core::state::StateBVLS<matrix_naive_type<double>>;

extern template class adelie_core::state::StateCSSCov<dense_type<float, Eigen::ColMajor>>;
extern template class adelie_core::state::StateCSSCov<dense_type<double, Eigen::ColMajor>>;

extern template class adelie_core::state::StateGaussianCov<constraint_type<float>, matrix_cov_type<float>>;
extern template class adelie_core::state::StateGaussianCov<constraint_type<double>, matrix_cov_type<double>>;

extern template class adelie_core::state::StateGaussianNaive<constraint_type<float>, matrix_naive_type<float>>;
extern template class adelie_core::state::StateGaussianNaive<constraint_type<double>, matrix_naive_type<double>>;

extern template class adelie_core::state::StateGaussianPinBase<constraint_type<float>>;
extern template class adelie_core::state::StateGaussianPinBase<constraint_type<double>>;

extern template class adelie_core::state::StateGaussianPinCov<constraint_type<float>, matrix_cov_type<float>>;
extern template class adelie_core::state::StateGaussianPinCov<constraint_type<double>, matrix_cov_type<double>>;

extern template class adelie_core::state::StateGaussianPinNaive<constraint_type<float>, matrix_naive_type<float>>;
extern template class adelie_core::state::StateGaussianPinNaive<constraint_type<double>, matrix_naive_type<double>>;

extern template class adelie_core::state::StateGlmNaive<constraint_type<float>, matrix_naive_type<float>>;
extern template class adelie_core::state::StateGlmNaive<constraint_type<double>, matrix_naive_type<double>>;

extern template class adelie_core::state::StateMultiGaussianNaive<constraint_type<float>, matrix_naive_type<float>>;
extern template class adelie_core::state::StateMultiGaussianNaive<constraint_type<double>, matrix_naive_type<double>>;

extern template class adelie_core::state::StateMultiGlmNaive<constraint_type<float>, matrix_naive_type<float>>;
extern template class adelie_core::state::StateMultiGlmNaive<constraint_type<double>, matrix_naive_type<double>>;

extern template class adelie_core::state::StatePinball<matrix_constraint_type<float>>;
extern template class adelie_core::state::StatePinball<matrix_constraint_type<double>>;