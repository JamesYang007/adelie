#pragma once
#include <adelie_core/matrix/matrix_constraint_sparse.hpp>
#include <adelie_core/matrix/utils.hpp>

namespace adelie_core {
namespace matrix {

template <class SparseType, class IndexType>
MatrixConstraintSparse<SparseType, IndexType>::MatrixConstraintSparse(
    size_t rows,
    size_t cols,
    size_t nnz,
    const Eigen::Ref<const vec_sp_index_t>& outer,
    const Eigen::Ref<const vec_sp_index_t>& inner,
    const Eigen::Ref<const vec_sp_value_t>& value,
    size_t n_threads
): 
    _mat(rows, cols, nnz, outer.data(), inner.data(), value.data()),
    _n_threads(n_threads),
    _buff(_n_threads)
{
    if (n_threads < 1) {
        throw util::adelie_core_error("n_threads must be >= 1.");
    }
}

template <class SparseType, class IndexType>
void
MatrixConstraintSparse<SparseType, IndexType>::rmmul(
    int j,
    const Eigen::Ref<const colmat_value_t>& Q,
    Eigen::Ref<vec_value_t> out
) 
{
    out.matrix() = _mat.row(j) * Q;
}

template <class SparseType, class IndexType>
typename MatrixConstraintSparse<SparseType, IndexType>::value_t
MatrixConstraintSparse<SparseType, IndexType>::rvmul(
    int j,
    const Eigen::Ref<const vec_value_t>& v
) 
{
    return _mat.row(j).dot(v.matrix());
}

template <class SparseType, class IndexType>
void
MatrixConstraintSparse<SparseType, IndexType>::rvtmul(
    int j,
    value_t v,
    Eigen::Ref<vec_value_t> out
) 
{
    out.matrix() += v * _mat.row(j);
}

template <class SparseType, class IndexType>
void
MatrixConstraintSparse<SparseType, IndexType>::mul(
    const Eigen::Ref<const vec_value_t>& v,
    Eigen::Ref<vec_value_t> out
) 
{
    out.matrix() = v.matrix() * _mat;
}

template <class SparseType, class IndexType>
void
MatrixConstraintSparse<SparseType, IndexType>::tmul(
    const Eigen::Ref<const vec_value_t>& v,
    Eigen::Ref<vec_value_t> out
) 
{
    const auto routine = [&](int k) {
        out[k] = _mat.row(k).dot(v.matrix());
    };
    if (_n_threads <= 1) {
        for (int k = 0; k < out.size(); ++k) routine(k);
    } else {
        #pragma omp parallel for schedule(static) num_threads(_n_threads)
        for (int k = 0; k < out.size(); ++k) routine(k);
    }
}

template <class SparseType, class IndexType>
void
MatrixConstraintSparse<SparseType, IndexType>::cov(
    const Eigen::Ref<const colmat_value_t>& Q,
    Eigen::Ref<colmat_value_t> out
) 
{
    Eigen::setNbThreads(_n_threads);
    out.noalias() = _mat * Q * _mat.transpose();
}

template <class SparseType, class IndexType>
int
MatrixConstraintSparse<SparseType, IndexType>::rows() const 
{
    return _mat.rows();
}

template <class SparseType, class IndexType>
int
MatrixConstraintSparse<SparseType, IndexType>::cols() const
{
    return _mat.cols();
}

template <class SparseType, class IndexType>
void
MatrixConstraintSparse<SparseType, IndexType>::sp_mul(
    const Eigen::Ref<const vec_index_t>& indices,
    const Eigen::Ref<const vec_value_t>& values,
    Eigen::Ref<vec_value_t> out
) 
{
    out.setZero();
    for (Eigen::Index i = 0; i < indices.size(); ++i) {
        out.matrix() += values[i] * _mat.row(indices[i]);
    }
}

} // namespace matrix
} // namespace adelie_core