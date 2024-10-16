#include <tools/eigen_wrap.hpp>
#include <tools/types.hpp>
#include <adelie_core/matrix/matrix_naive_kronecker_eye.ipp>

template class adelie_core::matrix::MatrixNaiveKroneckerEye<float>;
template class adelie_core::matrix::MatrixNaiveKroneckerEye<double>;
template class adelie_core::matrix::MatrixNaiveKroneckerEyeDense<dense_type<float, Eigen::RowMajor>>;
template class adelie_core::matrix::MatrixNaiveKroneckerEyeDense<dense_type<float, Eigen::ColMajor>>;
template class adelie_core::matrix::MatrixNaiveKroneckerEyeDense<dense_type<double, Eigen::RowMajor>>;
template class adelie_core::matrix::MatrixNaiveKroneckerEyeDense<dense_type<double, Eigen::ColMajor>>;