#include <tools/eigen_wrap.hpp>
#include <tools/types.hpp>
#include <adelie_core/state/state_multigaussian_naive.ipp>

template class adelie_core::state::StateMultiGaussianNaive<constraint_type<float>, matrix_naive_type<float>>;
template class adelie_core::state::StateMultiGaussianNaive<constraint_type<double>, matrix_naive_type<double>>;