#pragma once
#include <adelie_core/matrix/matrix_constraint_sparse.hpp>
#include <adelie_core/matrix/utils.hpp>

namespace adelie_core {
namespace matrix {

ADELIE_CORE_MATRIX_CONSTRAINT_SPARSE_TP
ADELIE_CORE_MATRIX_CONSTRAINT_SPARSE::MatrixConstraintSparse(
    size_t rows,
    size_t cols,
    size_t nnz,
    const Eigen::Ref<const vec_sp_index_t>& outer,
    const Eigen::Ref<const vec_sp_index_t>& inner,
    const Eigen::Ref<const vec_sp_value_t>& value,
    size_t n_threads
): 
    _mat(rows, cols, nnz, outer.data(), inner.data(), value.data()),
    _n_threads(n_threads)
{
    if (n_threads < 1) {
        throw util::adelie_core_error("n_threads must be >= 1.");
    }
}

ADELIE_CORE_MATRIX_CONSTRAINT_SPARSE_TP
void
ADELIE_CORE_MATRIX_CONSTRAINT_SPARSE::rmmul(
    int j,
    const Eigen::Ref<const colmat_value_t>& Q,
    Eigen::Ref<vec_value_t> out
) 
{
    out.matrix() = _mat.row(j) * Q;
}

ADELIE_CORE_MATRIX_CONSTRAINT_SPARSE_TP
void
ADELIE_CORE_MATRIX_CONSTRAINT_SPARSE::rmmul_safe(
    int j,
    const Eigen::Ref<const colmat_value_t>& Q,
    Eigen::Ref<vec_value_t> out
) const 
{
    out.matrix() = _mat.row(j) * Q;
}

ADELIE_CORE_MATRIX_CONSTRAINT_SPARSE_TP
typename ADELIE_CORE_MATRIX_CONSTRAINT_SPARSE::value_t
ADELIE_CORE_MATRIX_CONSTRAINT_SPARSE::rvmul(
    int j,
    const Eigen::Ref<const vec_value_t>& v
) 
{
    return _mat.row(j).dot(v.matrix());
}

ADELIE_CORE_MATRIX_CONSTRAINT_SPARSE_TP
typename ADELIE_CORE_MATRIX_CONSTRAINT_SPARSE::value_t
ADELIE_CORE_MATRIX_CONSTRAINT_SPARSE::rvmul_safe(
    int j,
    const Eigen::Ref<const vec_value_t>& v
) const 
{
    return _mat.row(j).dot(v.matrix());
}

ADELIE_CORE_MATRIX_CONSTRAINT_SPARSE_TP
void
ADELIE_CORE_MATRIX_CONSTRAINT_SPARSE::rvtmul(
    int j,
    value_t v,
    Eigen::Ref<vec_value_t> out
) 
{
    out.matrix() += v * _mat.row(j);
}

ADELIE_CORE_MATRIX_CONSTRAINT_SPARSE_TP
void
ADELIE_CORE_MATRIX_CONSTRAINT_SPARSE::mul(
    const Eigen::Ref<const vec_value_t>& v,
    Eigen::Ref<vec_value_t> out
) const 
{
    out.matrix() = v.matrix() * _mat;
}

ADELIE_CORE_MATRIX_CONSTRAINT_SPARSE_TP
void
ADELIE_CORE_MATRIX_CONSTRAINT_SPARSE::tmul(
    const Eigen::Ref<const vec_value_t>& v,
    Eigen::Ref<vec_value_t> out
) const
{
    const auto routine = [&](int k) {
        out[k] = _mat.row(k).dot(v.matrix());
    };
    util::omp_parallel_for(routine, 0, out.size(), _n_threads);
}

ADELIE_CORE_MATRIX_CONSTRAINT_SPARSE_TP
void
ADELIE_CORE_MATRIX_CONSTRAINT_SPARSE::cov(
    const Eigen::Ref<const colmat_value_t>& Q,
    Eigen::Ref<colmat_value_t> out
) const
{
    out.noalias() = _mat * Q * _mat.transpose();
}

ADELIE_CORE_MATRIX_CONSTRAINT_SPARSE_TP
int
ADELIE_CORE_MATRIX_CONSTRAINT_SPARSE::rows() const 
{
    return _mat.rows();
}

ADELIE_CORE_MATRIX_CONSTRAINT_SPARSE_TP
int
ADELIE_CORE_MATRIX_CONSTRAINT_SPARSE::cols() const
{
    return _mat.cols();
}

ADELIE_CORE_MATRIX_CONSTRAINT_SPARSE_TP
void
ADELIE_CORE_MATRIX_CONSTRAINT_SPARSE::sp_mul(
    const Eigen::Ref<const vec_index_t>& indices,
    const Eigen::Ref<const vec_value_t>& values,
    Eigen::Ref<vec_value_t> out
) const
{
    out.setZero();
    for (Eigen::Index i = 0; i < indices.size(); ++i) {
        out.matrix() += values[i] * _mat.row(indices[i]);
    }
}

} // namespace matrix
} // namespace adelie_core