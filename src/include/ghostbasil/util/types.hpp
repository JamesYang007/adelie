#pragma once
#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace ghostbasil {
namespace util {

template <class Scalar_, int Rows_=Eigen::Dynamic, int Cols_=Eigen::Dynamic>
using mat_type = Eigen::Matrix<Scalar_, Rows_, Cols_>;

template <class Scalar_, int Rows_=Eigen::Dynamic>
using vec_type = Eigen::Matrix<Scalar_, Rows_, 1>;

template <class Scalar_, int Options_=Eigen::ColMajor, class StorageIndex_=int>
using sp_vec_type = Eigen::SparseVector<Scalar_, Options_, StorageIndex_>;

} // namespace util
} // namespace ghostbasil
