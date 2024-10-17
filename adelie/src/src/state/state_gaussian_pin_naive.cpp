#include <tools/eigen_wrap.hpp>
#include <tools/types.hpp>
#include <adelie_core/state/state_gaussian_pin_naive.ipp>

template class adelie_core::state::StateGaussianPinNaive<constraint_type<float>, matrix_naive_type<float>>;
template class adelie_core::state::StateGaussianPinNaive<constraint_type<double>, matrix_naive_type<double>>;