#include <tools/eigen_wrap.hpp>
#include <tools/types.hpp>
#include <adelie_core/matrix/matrix_constraint_sparse.ipp>

template class adelie_core::matrix::MatrixConstraintSparse<sparse_type<float, Eigen::RowMajor>>;
template class adelie_core::matrix::MatrixConstraintSparse<sparse_type<double, Eigen::RowMajor>>;