#pragma once
#include <adelie_core/matrix/matrix_cov_sparse.hpp>
#include <adelie_core/matrix/utils.hpp>

namespace adelie_core {
namespace matrix {

ADELIE_CORE_MATRIX_COV_SPARSE_TP
ADELIE_CORE_MATRIX_COV_SPARSE::MatrixCovSparse(
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

ADELIE_CORE_MATRIX_COV_SPARSE_TP
void
ADELIE_CORE_MATRIX_COV_SPARSE::bmul(
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
        const auto outer = _mat.outerIndexPtr()[j];
        const auto size = _mat.outerIndexPtr()[j+1] - outer;
        const Eigen::Map<const vec_sp_index_t> inner(
            _mat.innerIndexPtr() + outer, size
        );
        const Eigen::Map<const vec_sp_value_t> value(
            _mat.valuePtr() + outer, size
        );
        out[j_idx] = svsvdot(indices, values, inner, value);
    }
}

ADELIE_CORE_MATRIX_COV_SPARSE_TP
void
ADELIE_CORE_MATRIX_COV_SPARSE::mul(
    const Eigen::Ref<const vec_index_t>& indices,
    const Eigen::Ref<const vec_value_t>& values,
    Eigen::Ref<vec_value_t> out
) const
{
    base_t::check_mul(indices.size(), values.size(), out.size(), rows(), cols());
    const auto routine = [&](int j) {
        const auto outer = _mat.outerIndexPtr()[j];
        const auto size = _mat.outerIndexPtr()[j+1] - outer;
        const Eigen::Map<const vec_sp_index_t> inner(
            _mat.innerIndexPtr() + outer, size
        );
        const Eigen::Map<const vec_sp_value_t> value(
            _mat.valuePtr() + outer, size
        );
        out[j] = svsvdot(indices, values, inner, value);
    };
    util::omp_parallel_for(routine, 0, _mat.cols(), _n_threads);
} 

ADELIE_CORE_MATRIX_COV_SPARSE_TP
void
ADELIE_CORE_MATRIX_COV_SPARSE::to_dense(
    int i, int p,
    Eigen::Ref<colmat_value_t> out
) const
{
    base_t::check_to_dense(i, p, out.rows(), out.cols(), rows(), cols());
    out = _mat.block(i, i, p, p);
}

ADELIE_CORE_MATRIX_COV_SPARSE_TP
int
ADELIE_CORE_MATRIX_COV_SPARSE::cols() const
{
    return _mat.cols();
}

} // namespace matrix
} // namespace adelie_core