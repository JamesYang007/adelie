#include <tools/eigen_wrap.hpp>
#include <adelie_core/constraint/elastic_net/constraint_linear.ipp>
#include <adelie_core/matrix/matrix_constraint_base.hpp>

template class adelie_core::constraint::ConstraintLinear<adelie_core::matrix::MatrixConstraintBase<float>>;
template class adelie_core::constraint::ConstraintLinear<adelie_core::matrix::MatrixConstraintBase<double>>;