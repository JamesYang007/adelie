#include <tools/eigen_wrap.hpp>
#include <tools/types.hpp>
#include <adelie_core/state/state_gaussian_pin_base.ipp>

template class adelie_core::state::StateGaussianPinBase<constraint_type<float>>;
template class adelie_core::state::StateGaussianPinBase<constraint_type<double>>;
template class adelie_core::state::StateGaussianPinBase<constraint_type<float>, float, Eigen::Index, int8_t>;
template class adelie_core::state::StateGaussianPinBase<constraint_type<double>, double, Eigen::Index, int8_t>;