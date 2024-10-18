#include <tools/eigen_wrap.hpp>
#include <tools/types.hpp>
#include <adelie_core/matrix/matrix_naive_convex_relu.ipp>

template class adelie_core::matrix::MatrixNaiveConvexReluDense<dense_type<float, Eigen::RowMajor>, dense_type<bool, Eigen::ColMajor>>;
template class adelie_core::matrix::MatrixNaiveConvexReluDense<dense_type<float, Eigen::ColMajor>, dense_type<bool, Eigen::ColMajor>>;
template class adelie_core::matrix::MatrixNaiveConvexReluDense<dense_type<double, Eigen::RowMajor>, dense_type<bool, Eigen::ColMajor>>;
template class adelie_core::matrix::MatrixNaiveConvexReluDense<dense_type<double, Eigen::ColMajor>, dense_type<bool, Eigen::ColMajor>>;
template class adelie_core::matrix::MatrixNaiveConvexReluSparse<sparse_type<float, Eigen::ColMajor>, dense_type<bool, Eigen::ColMajor>>;
template class adelie_core::matrix::MatrixNaiveConvexReluSparse<sparse_type<double, Eigen::ColMajor>, dense_type<bool, Eigen::ColMajor>>;