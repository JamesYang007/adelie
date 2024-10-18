#pragma once
#include <adelie_core/matrix/matrix_constraint_dense.hpp>
#include <adelie_core/matrix/utils.hpp>

namespace adelie_core {
namespace matrix {

ADELIE_CORE_MATRIX_CONSTRAINT_DENSE_TP
ADELIE_CORE_MATRIX_CONSTRAINT_DENSE::MatrixConstraintDense(
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

ADELIE_CORE_MATRIX_CONSTRAINT_DENSE_TP
void
ADELIE_CORE_MATRIX_CONSTRAINT_DENSE::rmmul(
    int j,
    const Eigen::Ref<const colmat_value_t>& Q,
    Eigen::Ref<vec_value_t> out
) 
{
    out.matrix() = _mat.row(j) * Q;
}

ADELIE_CORE_MATRIX_CONSTRAINT_DENSE_TP
typename ADELIE_CORE_MATRIX_CONSTRAINT_DENSE::value_t
ADELIE_CORE_MATRIX_CONSTRAINT_DENSE::rvmul(
    int j,
    const Eigen::Ref<const vec_value_t>& v
) 
{
    return _mat.row(j).dot(v.matrix());
}

ADELIE_CORE_MATRIX_CONSTRAINT_DENSE_TP
void
ADELIE_CORE_MATRIX_CONSTRAINT_DENSE::rvtmul(
    int j,
    value_t v,
    Eigen::Ref<vec_value_t> out
) 
{
    out.matrix() += v * _mat.row(j);
}

ADELIE_CORE_MATRIX_CONSTRAINT_DENSE_TP
void
ADELIE_CORE_MATRIX_CONSTRAINT_DENSE::mul(
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

ADELIE_CORE_MATRIX_CONSTRAINT_DENSE_TP
void
ADELIE_CORE_MATRIX_CONSTRAINT_DENSE::tmul(
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

ADELIE_CORE_MATRIX_CONSTRAINT_DENSE_TP
void
ADELIE_CORE_MATRIX_CONSTRAINT_DENSE::cov(
    const Eigen::Ref<const colmat_value_t>& Q,
    Eigen::Ref<colmat_value_t> out
) 
{
    Eigen::setNbThreads(_n_threads);
    out.noalias() = _mat * Q * _mat.transpose();
}

ADELIE_CORE_MATRIX_CONSTRAINT_DENSE_TP
int
ADELIE_CORE_MATRIX_CONSTRAINT_DENSE::rows() const
{
    return _mat.rows();
}

ADELIE_CORE_MATRIX_CONSTRAINT_DENSE_TP
int
ADELIE_CORE_MATRIX_CONSTRAINT_DENSE::cols() const 
{
    return _mat.cols();
}

ADELIE_CORE_MATRIX_CONSTRAINT_DENSE_TP
void
ADELIE_CORE_MATRIX_CONSTRAINT_DENSE::sp_mul(
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