#include <tools/eigen_wrap.hpp>
#include <tools/types.hpp>
#include <adelie_core/state/state_base.ipp>

template class adelie_core::state::StateBase<constraint_type<float>>;
template class adelie_core::state::StateBase<constraint_type<double>>;