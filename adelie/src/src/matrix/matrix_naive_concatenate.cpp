#include <tools/eigen_wrap.hpp>
#include <adelie_core/matrix/matrix_naive_concatenate.ipp>

template class adelie_core::matrix::MatrixNaiveCConcatenate<float>;
template class adelie_core::matrix::MatrixNaiveCConcatenate<double>;
template class adelie_core::matrix::MatrixNaiveRConcatenate<float>;
template class adelie_core::matrix::MatrixNaiveRConcatenate<double>;