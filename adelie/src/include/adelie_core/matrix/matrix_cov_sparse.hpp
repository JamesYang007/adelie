#pragma once
#include <adelie_core/matrix/matrix_cov_base.hpp>

#ifndef ADELIE_CORE_MATRIX_COV_SPARSE_TP
#define ADELIE_CORE_MATRIX_COV_SPARSE_TP \
    template <class SparseType, class IndexType>
#endif
#ifndef ADELIE_CORE_MATRIX_COV_SPARSE
#define ADELIE_CORE_MATRIX_COV_SPARSE \
    MatrixCovSparse<SparseType, IndexType>
#endif

namespace adelie_core {
namespace matrix {

template <class SparseType, class IndexType=Eigen::Index>
class MatrixCovSparse: public MatrixCovBase<typename SparseType::Scalar, IndexType>
{
public:
    using base_t = MatrixCovBase<typename SparseType::Scalar, IndexType>;
    using sparse_t = SparseType;
    using typename base_t::value_t;
    using typename base_t::vec_index_t;
    using typename base_t::vec_value_t;
    using typename base_t::colmat_value_t;
    using vec_sp_value_t = vec_value_t;
    using vec_sp_index_t = util::rowvec_type<typename sparse_t::StorageIndex>;

    static_assert(!sparse_t::IsRowMajor, "MatrixCovSparse: only column-major allowed!");
    
private:
    const Eigen::Map<const sparse_t> _mat;  // underlying sparse matrix
    const size_t _n_threads;                // number of threads
    
public:
    explicit MatrixCovSparse(
        size_t rows,
        size_t cols,
        size_t nnz,
        const Eigen::Ref<const vec_sp_index_t>& outer,
        const Eigen::Ref<const vec_sp_index_t>& inner,
        const Eigen::Ref<const vec_sp_value_t>& value,
        size_t n_threads
    );

    using base_t::rows;
    
    ADELIE_CORE_MATRIX_COV_PURE_OVERRIDE_DECL    
};

} // namespace matrix
} // namespace adelie_core