#include <tools/eigen_wrap.hpp>
#include <tools/types.hpp>
#include <adelie_core/state/state_bvls.ipp>

template class adelie_core::state::StateBVLS<matrix_naive_type<float>>;
template class adelie_core::state::StateBVLS<matrix_naive_type<double>>;