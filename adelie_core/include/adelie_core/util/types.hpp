#pragma once
#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace adelie_core {
namespace util {
    
template <class Scalar_, int Rows_=Eigen::Dynamic, int Cols_=Eigen::Dynamic>
using colmat_type = Eigen::Matrix<Scalar_, Rows_, Cols_, Eigen::ColMajor>;

template <class Scalar_, int Rows_=Eigen::Dynamic, int Cols_=Eigen::Dynamic>
using rowmat_type = Eigen::Matrix<Scalar_, Rows_, Cols_, Eigen::RowMajor>;

template <class Scalar_, int Rows_=Eigen::Dynamic, int Cols_=Eigen::Dynamic>
using colarr_type = Eigen::Array<Scalar_, Rows_, Cols_, Eigen::ColMajor>;

template <class Scalar_, int Rows_=Eigen::Dynamic, int Cols_=Eigen::Dynamic>
using rowarr_type = Eigen::Array<Scalar_, Rows_, Cols_, Eigen::RowMajor>;

template <class Scalar_, int Rows_=Eigen::Dynamic>
using colvec_type = colarr_type<Scalar_, Rows_, 1>;

template <class Scalar_, int Cols_=Eigen::Dynamic>
using rowvec_type = rowarr_type<Scalar_, 1, Cols_>;

template <class Scalar_, int Options_=Eigen::ColMajor, class StorageIndex_=int>
using sp_vec_type = Eigen::SparseVector<Scalar_, Options_, StorageIndex_>;

} // namespace util
} // namespace adelie_core
