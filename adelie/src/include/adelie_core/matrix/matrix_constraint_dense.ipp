#pragma once
#include <adelie_core/matrix/matrix_constraint_dense.hpp>
#include <adelie_core/matrix/utils.hpp>

namespace adelie_core {
namespace matrix {

template <class DenseType, class IndexType>
MatrixConstraintDense<DenseType, IndexType>::MatrixConstraintDense(
    const Eigen::Ref<const dense_t>& mat,
    size_t n_threads
): 
    _mat(mat.data(), mat.rows(), mat.cols()),
    _n_threads(n_threads),
    _buff(_n_threads, std::min<size_t>(mat.rows(), mat.cols()))
{
    if (n_threads < 1) {
        throw util::adelie_core_error("n_threads must be >= 1.");
    }
}

template <class DenseType, class IndexType>
void
MatrixConstraintDense<DenseType, IndexType>::rmmul(
    int j,
    const Eigen::Ref<const colmat_value_t>& Q,
    Eigen::Ref<vec_value_t> out
) 
{
    out.matrix() = _mat.row(j) * Q;
}

template <class DenseType, class IndexType>
typename MatrixConstraintDense<DenseType, IndexType>::value_t
MatrixConstraintDense<DenseType, IndexType>::rvmul(
    int j,
    const Eigen::Ref<const vec_value_t>& v
) 
{
    return _mat.row(j).dot(v.matrix());
}

template <class DenseType, class IndexType>
void
MatrixConstraintDense<DenseType, IndexType>::rvtmul(
    int j,
    value_t v,
    Eigen::Ref<vec_value_t> out
) 
{
    out.matrix() += v * _mat.row(j);
}

template <class DenseType, class IndexType>
void
MatrixConstraintDense<DenseType, IndexType>::mul(
    const Eigen::Ref<const vec_value_t>& v,
    Eigen::Ref<vec_value_t> out
) 
{
    auto out_m = out.matrix();
    dgemv(
        _mat,
        v.matrix(),
        _n_threads,
        _buff,
        out_m
    );
}

template <class DenseType, class IndexType>
void
MatrixConstraintDense<DenseType, IndexType>::tmul(
    const Eigen::Ref<const vec_value_t>& v,
    Eigen::Ref<vec_value_t> out
) 
{
    auto out_m = out.matrix();
    dgemv(
        _mat.transpose(),
        v.matrix(),
        _n_threads,
        _buff,
        out_m
    );
}

template <class DenseType, class IndexType>
void
MatrixConstraintDense<DenseType, IndexType>::cov(
    const Eigen::Ref<const colmat_value_t>& Q,
    Eigen::Ref<colmat_value_t> out
) 
{
    Eigen::setNbThreads(_n_threads);
    out.noalias() = _mat * Q * _mat.transpose();
}

template <class DenseType, class IndexType>
int
MatrixConstraintDense<DenseType, IndexType>::rows() const
{
    return _mat.rows();
}

template <class DenseType, class IndexType>
int
MatrixConstraintDense<DenseType, IndexType>::cols() const 
{
    return _mat.cols();
}

template <class DenseType, class IndexType>
void
MatrixConstraintDense<DenseType, IndexType>::sp_mul(
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