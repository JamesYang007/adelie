#include <tools/eigen_wrap.hpp>
#include <tools/types.hpp>
#include <adelie_core/state/state_gaussian_pin_cov.ipp>

template class adelie_core::state::StateGaussianPinCov<constraint_type<float>, matrix_cov_type<float>>;
template class adelie_core::state::StateGaussianPinCov<constraint_type<double>, matrix_cov_type<double>>;