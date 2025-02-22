#include <tools/eigen_wrap.hpp>
#include <tools/types.hpp>
#include <adelie_core/state/state_pinball.ipp>

template class adelie_core::state::StatePinball<matrix_constraint_type<float>>;
template class adelie_core::state::StatePinball<matrix_constraint_type<double>>;