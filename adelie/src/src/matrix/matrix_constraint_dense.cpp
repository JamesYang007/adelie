#include <tools/eigen_wrap.hpp>
#include <tools/types.hpp>
#include <adelie_core/matrix/matrix_constraint_dense.ipp>

template class adelie_core::matrix::MatrixConstraintDense<dense_type<float, Eigen::RowMajor>>;
template class adelie_core::matrix::MatrixConstraintDense<dense_type<float, Eigen::ColMajor>>;
template class adelie_core::matrix::MatrixConstraintDense<dense_type<double, Eigen::RowMajor>>;
template class adelie_core::matrix::MatrixConstraintDense<dense_type<double, Eigen::ColMajor>>;