#pragma once
#include <adelie_core/constraint/elastic_net/constraint_base.hpp>
#include <adelie_core/constraint/elastic_net/constraint_box.hpp>
#include <adelie_core/constraint/elastic_net/constraint_linear.hpp>
#include <adelie_core/constraint/elastic_net/constraint_one_sided.hpp>
#include <adelie_core/matrix/matrix_constraint_base.hpp>

extern template class adelie_core::constraint::ConstraintBase<float>;
extern template class adelie_core::constraint::ConstraintBase<double>;

extern template class adelie_core::constraint::ConstraintBox<float>;
extern template class adelie_core::constraint::ConstraintBox<double>;

extern template class adelie_core::constraint::ConstraintLinear<adelie_core::matrix::MatrixConstraintBase<float>>;
extern template class adelie_core::constraint::ConstraintLinear<adelie_core::matrix::MatrixConstraintBase<double>>;

extern template class adelie_core::constraint::ConstraintOneSided<float>;
extern template class adelie_core::constraint::ConstraintOneSided<double>;
extern template class adelie_core::constraint::ConstraintOneSidedADMM<float>;
extern template class adelie_core::constraint::ConstraintOneSidedADMM<double>;