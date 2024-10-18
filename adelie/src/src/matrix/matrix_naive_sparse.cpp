#include <tools/eigen_wrap.hpp>
#include <tools/types.hpp>
#include <adelie_core/matrix/matrix_naive_sparse.ipp>

template class adelie_core::matrix::MatrixNaiveSparse<sparse_type<float, Eigen::ColMajor>>;
template class adelie_core::matrix::MatrixNaiveSparse<sparse_type<double, Eigen::ColMajor>>;