#include <tools/eigen_wrap.hpp>
#include <tools/types.hpp>
#include <adelie_core/matrix/matrix_naive_one_hot.ipp>

template class adelie_core::matrix::MatrixNaiveOneHotDense<dense_type<float, Eigen::RowMajor>>;
template class adelie_core::matrix::MatrixNaiveOneHotDense<dense_type<float, Eigen::ColMajor>>;
template class adelie_core::matrix::MatrixNaiveOneHotDense<dense_type<double, Eigen::RowMajor>>;
template class adelie_core::matrix::MatrixNaiveOneHotDense<dense_type<double, Eigen::ColMajor>>;