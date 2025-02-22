#include <tools/eigen_wrap.hpp>
#include <adelie_core/constraint/elastic_net/constraint_one_sided.ipp>

template class adelie_core::constraint::ConstraintOneSided<float>;
template class adelie_core::constraint::ConstraintOneSided<double>;
template class adelie_core::constraint::ConstraintOneSidedADMM<float>;
template class adelie_core::constraint::ConstraintOneSidedADMM<double>;