#pragma once
#include <tools/types.hpp>
#include <adelie_core/matrix/matrix_constraint_base.hpp>
#include <adelie_core/matrix/matrix_constraint_dense.hpp>
#include <adelie_core/matrix/matrix_constraint_sparse.hpp>
#include <adelie_core/matrix/matrix_cov_base.hpp>
#include <adelie_core/matrix/matrix_cov_block_diag.hpp>
#include <adelie_core/matrix/matrix_cov_dense.hpp>
#include <adelie_core/matrix/matrix_cov_lazy_cov.hpp>
#include <adelie_core/matrix/matrix_cov_sparse.hpp>
#include <adelie_core/matrix/matrix_naive_base.hpp>
#include <adelie_core/matrix/matrix_naive_block_diag.hpp>
#include <adelie_core/matrix/matrix_naive_concatenate.hpp>
#include <adelie_core/matrix/matrix_naive_convex_gated_relu.hpp>
#include <adelie_core/matrix/matrix_naive_convex_relu.hpp>
#include <adelie_core/matrix/matrix_naive_dense.hpp>
#include <adelie_core/matrix/matrix_naive_interaction.hpp>
#include <adelie_core/matrix/matrix_naive_kronecker_eye.hpp>
#include <adelie_core/matrix/matrix_naive_one_hot.hpp>
#include <adelie_core/matrix/matrix_naive_snp_phased_ancestry.hpp>
#include <adelie_core/matrix/matrix_naive_snp_unphased.hpp>
#include <adelie_core/matrix/matrix_naive_sparse.hpp>
#include <adelie_core/matrix/matrix_naive_standardize.hpp>
#include <adelie_core/matrix/matrix_naive_subset.hpp>

extern template class adelie_core::matrix::MatrixConstraintBase<float>;
extern template class adelie_core::matrix::MatrixConstraintBase<double>;

extern template class adelie_core::matrix::MatrixConstraintDense<dense_type<float, Eigen::RowMajor>>;
extern template class adelie_core::matrix::MatrixConstraintDense<dense_type<float, Eigen::ColMajor>>;
extern template class adelie_core::matrix::MatrixConstraintDense<dense_type<double, Eigen::RowMajor>>;
extern template class adelie_core::matrix::MatrixConstraintDense<dense_type<double, Eigen::ColMajor>>;

extern template class adelie_core::matrix::MatrixConstraintSparse<sparse_type<float, Eigen::RowMajor>>;
extern template class adelie_core::matrix::MatrixConstraintSparse<sparse_type<double, Eigen::RowMajor>>;

extern template class adelie_core::matrix::MatrixCovBase<float>;
extern template class adelie_core::matrix::MatrixCovBase<double>;

extern template class adelie_core::matrix::MatrixCovBlockDiag<float>;
extern template class adelie_core::matrix::MatrixCovBlockDiag<double>;

extern template class adelie_core::matrix::MatrixCovDense<dense_type<float, Eigen::RowMajor>>;
extern template class adelie_core::matrix::MatrixCovDense<dense_type<float, Eigen::ColMajor>>;
extern template class adelie_core::matrix::MatrixCovDense<dense_type<double, Eigen::RowMajor>>;
extern template class adelie_core::matrix::MatrixCovDense<dense_type<double, Eigen::ColMajor>>;

extern template class adelie_core::matrix::MatrixCovLazyCov<dense_type<float, Eigen::RowMajor>>;
extern template class adelie_core::matrix::MatrixCovLazyCov<dense_type<float, Eigen::ColMajor>>;
extern template class adelie_core::matrix::MatrixCovLazyCov<dense_type<double, Eigen::RowMajor>>;
extern template class adelie_core::matrix::MatrixCovLazyCov<dense_type<double, Eigen::ColMajor>>;

extern template class adelie_core::matrix::MatrixCovSparse<sparse_type<float, Eigen::ColMajor>>;
extern template class adelie_core::matrix::MatrixCovSparse<sparse_type<double, Eigen::ColMajor>>;

extern template class adelie_core::matrix::MatrixNaiveBase<float>;
extern template class adelie_core::matrix::MatrixNaiveBase<double>;

extern template class adelie_core::matrix::MatrixNaiveBlockDiag<float>;
extern template class adelie_core::matrix::MatrixNaiveBlockDiag<double>;

extern template class adelie_core::matrix::MatrixNaiveCConcatenate<float>;
extern template class adelie_core::matrix::MatrixNaiveCConcatenate<double>;
extern template class adelie_core::matrix::MatrixNaiveRConcatenate<float>;
extern template class adelie_core::matrix::MatrixNaiveRConcatenate<double>;

extern template class adelie_core::matrix::MatrixNaiveConvexGatedReluDense<dense_type<float, Eigen::RowMajor>, dense_type<bool, Eigen::ColMajor>>;
extern template class adelie_core::matrix::MatrixNaiveConvexGatedReluDense<dense_type<float, Eigen::ColMajor>, dense_type<bool, Eigen::ColMajor>>;
extern template class adelie_core::matrix::MatrixNaiveConvexGatedReluDense<dense_type<double, Eigen::RowMajor>, dense_type<bool, Eigen::ColMajor>>;
extern template class adelie_core::matrix::MatrixNaiveConvexGatedReluDense<dense_type<double, Eigen::ColMajor>, dense_type<bool, Eigen::ColMajor>>;
extern template class adelie_core::matrix::MatrixNaiveConvexGatedReluSparse<sparse_type<float, Eigen::ColMajor>, dense_type<bool, Eigen::ColMajor>>;
extern template class adelie_core::matrix::MatrixNaiveConvexGatedReluSparse<sparse_type<double, Eigen::ColMajor>, dense_type<bool, Eigen::ColMajor>>;

extern template class adelie_core::matrix::MatrixNaiveConvexReluDense<dense_type<float, Eigen::RowMajor>, dense_type<bool, Eigen::ColMajor>>;
extern template class adelie_core::matrix::MatrixNaiveConvexReluDense<dense_type<float, Eigen::ColMajor>, dense_type<bool, Eigen::ColMajor>>;
extern template class adelie_core::matrix::MatrixNaiveConvexReluDense<dense_type<double, Eigen::RowMajor>, dense_type<bool, Eigen::ColMajor>>;
extern template class adelie_core::matrix::MatrixNaiveConvexReluDense<dense_type<double, Eigen::ColMajor>, dense_type<bool, Eigen::ColMajor>>;
extern template class adelie_core::matrix::MatrixNaiveConvexReluSparse<sparse_type<float, Eigen::ColMajor>, dense_type<bool, Eigen::ColMajor>>;
extern template class adelie_core::matrix::MatrixNaiveConvexReluSparse<sparse_type<double, Eigen::ColMajor>, dense_type<bool, Eigen::ColMajor>>;

extern template class adelie_core::matrix::MatrixNaiveDense<dense_type<float, Eigen::RowMajor>>;
extern template class adelie_core::matrix::MatrixNaiveDense<dense_type<float, Eigen::ColMajor>>;
extern template class adelie_core::matrix::MatrixNaiveDense<dense_type<double, Eigen::RowMajor>>;
extern template class adelie_core::matrix::MatrixNaiveDense<dense_type<double, Eigen::ColMajor>>;

extern template class adelie_core::matrix::MatrixNaiveInteractionDense<dense_type<float, Eigen::RowMajor>>;
extern template class adelie_core::matrix::MatrixNaiveInteractionDense<dense_type<float, Eigen::ColMajor>>;
extern template class adelie_core::matrix::MatrixNaiveInteractionDense<dense_type<double, Eigen::RowMajor>>;
extern template class adelie_core::matrix::MatrixNaiveInteractionDense<dense_type<double, Eigen::ColMajor>>;

extern template class adelie_core::matrix::MatrixNaiveKroneckerEye<float>;
extern template class adelie_core::matrix::MatrixNaiveKroneckerEye<double>;
extern template class adelie_core::matrix::MatrixNaiveKroneckerEyeDense<dense_type<float, Eigen::RowMajor>>;
extern template class adelie_core::matrix::MatrixNaiveKroneckerEyeDense<dense_type<float, Eigen::ColMajor>>;
extern template class adelie_core::matrix::MatrixNaiveKroneckerEyeDense<dense_type<double, Eigen::RowMajor>>;
extern template class adelie_core::matrix::MatrixNaiveKroneckerEyeDense<dense_type<double, Eigen::ColMajor>>;

extern template class adelie_core::matrix::MatrixNaiveOneHotDense<dense_type<float, Eigen::RowMajor>>;
extern template class adelie_core::matrix::MatrixNaiveOneHotDense<dense_type<float, Eigen::ColMajor>>;
extern template class adelie_core::matrix::MatrixNaiveOneHotDense<dense_type<double, Eigen::RowMajor>>;
extern template class adelie_core::matrix::MatrixNaiveOneHotDense<dense_type<double, Eigen::ColMajor>>;

extern template class adelie_core::matrix::MatrixNaiveSNPPhasedAncestry<float>;
extern template class adelie_core::matrix::MatrixNaiveSNPPhasedAncestry<double>;

extern template class adelie_core::matrix::MatrixNaiveSNPUnphased<float>;
extern template class adelie_core::matrix::MatrixNaiveSNPUnphased<double>;

extern template class adelie_core::matrix::MatrixNaiveSparse<sparse_type<float, Eigen::ColMajor>>;
extern template class adelie_core::matrix::MatrixNaiveSparse<sparse_type<double, Eigen::ColMajor>>;

extern template class adelie_core::matrix::MatrixNaiveStandardize<float>;
extern template class adelie_core::matrix::MatrixNaiveStandardize<double>;

extern template class adelie_core::matrix::MatrixNaiveCSubset<float>;
extern template class adelie_core::matrix::MatrixNaiveCSubset<double>;
extern template class adelie_core::matrix::MatrixNaiveRSubset<float>;
extern template class adelie_core::matrix::MatrixNaiveRSubset<double>;