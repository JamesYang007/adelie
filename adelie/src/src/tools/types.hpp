#pragma once

template <class T, int Storage>
using dense_type = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Storage>;

template <class T, int Storage>
using sparse_type = Eigen::SparseMatrix<T, Storage>;