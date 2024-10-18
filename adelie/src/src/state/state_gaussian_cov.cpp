#include <tools/eigen_wrap.hpp>
#include <tools/types.hpp>
#include <adelie_core/state/state_gaussian_cov.ipp>

template class adelie_core::state::StateGaussianCov<constraint_type<float>, matrix_cov_type<float>>;
template class adelie_core::state::StateGaussianCov<constraint_type<double>, matrix_cov_type<double>>;