#pragma once
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <adelie_core/constraint/constraint_base.hpp>
#include <adelie_core/glm/glm_base.hpp>
#include <adelie_core/glm/glm_multibase.hpp>
#include <adelie_core/matrix/matrix_constraint_base.hpp>
#include <adelie_core/matrix/matrix_cov_base.hpp>
#include <adelie_core/matrix/matrix_naive_base.hpp>

template <class T, int Storage>
using dense_type = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Storage>;

template <class T, int Storage>
using sparse_type = Eigen::SparseMatrix<T, Storage>;

template <class T>
using constraint_type = adelie_core::constraint::ConstraintBase<T>;

template <class T>
using glm_type = adelie_core::glm::GlmBase<T>;

template <class T>
using glm_multi_type = adelie_core::glm::GlmMultiBase<T>;

template <class T>
using matrix_constraint_type = adelie_core::matrix::MatrixConstraintBase<T>;

template <class T>
using matrix_cov_type = adelie_core::matrix::MatrixCovBase<T>;

template <class T>
using matrix_naive_type = adelie_core::matrix::MatrixNaiveBase<T>;