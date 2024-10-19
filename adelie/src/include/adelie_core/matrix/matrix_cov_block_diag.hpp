#pragma once
#include <adelie_core/matrix/matrix_cov_base.hpp>

#ifndef ADELIE_CORE_MATRIX_COV_BLOCK_DIAG_TP
#define ADELIE_CORE_MATRIX_COV_BLOCK_DIAG_TP \
    template <class ValueType, class IndexType>
#endif
#ifndef ADELIE_CORE_MATRIX_COV_BLOCK_DIAG
#define ADELIE_CORE_MATRIX_COV_BLOCK_DIAG \
    MatrixCovBlockDiag<ValueType, IndexType>
#endif

namespace adelie_core {
namespace matrix {

template <class ValueType, class IndexType=Eigen::Index>
class MatrixCovBlockDiag: public MatrixCovBase<ValueType, IndexType>
{
public:
    using base_t = MatrixCovBase<ValueType, IndexType>;
    using typename base_t::value_t;
    using typename base_t::vec_index_t;
    using typename base_t::vec_value_t;
    using typename base_t::colmat_value_t;
    
private:
    const std::vector<base_t*> _mat_list;   // (L,) list of covariance matrices
    const vec_index_t _mat_size_cumsum;     // (L+1,) list of matrix size cumulative sum.
    const size_t _cols;                     // number of columns
    const vec_index_t _slice_map;           // (p,) array mapping to matrix slice
    const size_t _n_threads;                // number of threads
    vec_index_t _ibuff;
    vec_value_t _vbuff;

    static inline auto init_mat_size_cumsum(
        const std::vector<base_t*>& mat_list
    );

    static inline auto init_cols(
        const std::vector<base_t*>& mat_list
    );

    static inline auto init_slice_map(
        const std::vector<base_t*>& mat_list,
        size_t p
    );

public:
    explicit MatrixCovBlockDiag(
        const std::vector<base_t*>& mat_list,
        size_t n_threads
    );

    using base_t::rows;

    ADELIE_CORE_MATRIX_COV_PURE_OVERRIDE_DECL    
};

} // namespace matrix
} // namespace adelie_core