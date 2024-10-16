#pragma once
#include <adelie_core/matrix/matrix_cov_dense.hpp>
#include <adelie_core/matrix/utils.hpp>

namespace adelie_core {
namespace matrix {

template <class DenseType, class IndexType>
MatrixCovDense<DenseType, IndexType>::MatrixCovDense(
    const Eigen::Ref<const dense_t>& mat,
    size_t n_threads
): 
    _mat(mat.data(), mat.rows(), mat.cols()),
    _n_threads(n_threads)
{
    if (mat.rows() != mat.cols()) {
        throw util::adelie_core_error("mat must be (p, p).");
    }
    if (n_threads < 1) {
        throw util::adelie_core_error("n_threads must be >= 1.");
    }
}

template <class DenseType, class IndexType>
void
MatrixCovDense<DenseType, IndexType>::bmul(
    const Eigen::Ref<const vec_index_t>& subset,
    const Eigen::Ref<const vec_index_t>& indices,
    const Eigen::Ref<const vec_value_t>& values,
    Eigen::Ref<vec_value_t> out
) 
{
    base_t::check_bmul(subset.size(), indices.size(), values.size(), out.size(), rows(), cols());
    out.setZero();
    for (int j_idx = 0; j_idx < subset.size(); ++j_idx) {
        const auto j = subset[j_idx];
        for (int i_idx = 0; i_idx < indices.size(); ++i_idx) {
            const auto i = indices[i_idx];
            const auto v = values[i_idx];
            out[j_idx] += v * _mat(i, j);
        }
    }
}

template <class DenseType, class IndexType>
void
MatrixCovDense<DenseType, IndexType>::mul(
    const Eigen::Ref<const vec_index_t>& indices,
    const Eigen::Ref<const vec_value_t>& values,
    Eigen::Ref<vec_value_t> out
) 
{
    base_t::check_mul(indices.size(), values.size(), out.size(), rows(), cols());
    out.setZero();
    for (int i_idx = 0; i_idx < indices.size(); ++i_idx) {
        const auto i = indices[i_idx];
        const auto v = values[i_idx];
        if constexpr (dense_t::IsRowMajor) {
            dvaddi(out, v * _mat.row(i).array(), _n_threads);
        } else {
            dvaddi(out, v * _mat.col(i).array(), _n_threads);
        }
    }
} 

template <class DenseType, class IndexType>
void
MatrixCovDense<DenseType, IndexType>::to_dense(
    int i, int p,
    Eigen::Ref<colmat_value_t> out
) 
{
    base_t::check_to_dense(i, p, out.rows(), out.cols(), rows(), cols());
    out = _mat.block(i, i, p, p);
}

template <class DenseType, class IndexType>
int
MatrixCovDense<DenseType, IndexType>::cols() const
{
    return _mat.cols();
}

} // namespace matrix
} // namespace adelie_core