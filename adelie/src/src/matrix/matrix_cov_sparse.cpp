#include <tools/eigen_wrap.hpp>
#include <tools/types.hpp>
#include <adelie_core/matrix/matrix_cov_sparse.ipp>

template class adelie_core::matrix::MatrixCovSparse<sparse_type<float, Eigen::ColMajor>>;
template class adelie_core::matrix::MatrixCovSparse<sparse_type<double, Eigen::ColMajor>>;