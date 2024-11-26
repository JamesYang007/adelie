#pragma once
#include <adelie_core/matrix/matrix_naive_base.hpp>

#ifndef ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG_TP
#define ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG_TP \
    template <class ValueType, class IndexType>
#endif
#ifndef ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG
#define ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG \
    MatrixNaiveBlockDiag<ValueType, IndexType>
#endif

namespace adelie_core {
namespace matrix {

template <class ValueType, class IndexType=Eigen::Index> 
class MatrixNaiveBlockDiag: public MatrixNaiveBase<ValueType, IndexType>
{
public:
    using base_t = MatrixNaiveBase<ValueType, IndexType>;
    using typename base_t::index_t;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::colmat_value_t;
    using typename base_t::rowmat_value_t;
    using typename base_t::sp_mat_value_t;

private:
    const std::vector<base_t*> _mat_list;
    const size_t _rows;
    const size_t _cols;
    const size_t _max_cols;
    const vec_index_t _col_slice_map;
    const vec_index_t _col_index_map;
    const vec_index_t _row_outer;
    const vec_index_t _col_outer;
    const size_t _n_threads;

    static inline auto init_rows(
        const std::vector<base_t*>& mat_list
    );

    static inline auto init_cols(
        const std::vector<base_t*>& mat_list
    );

    static inline auto init_max_cols(
        const std::vector<base_t*>& mat_list
    );

    static inline auto init_col_slice_map(
        const std::vector<base_t*>& mat_list,
        size_t p
    );

    static inline auto init_col_index_map(
        const std::vector<base_t*>& mat_list,
        size_t p
    );

    static inline auto init_row_outer(
        const std::vector<base_t*>& mat_list
    );

    static inline auto init_col_outer(
        const std::vector<base_t*>& mat_list
    );

public:
    explicit MatrixNaiveBlockDiag(
        const std::vector<base_t*>& mat_list,
        size_t n_threads
    );

    ADELIE_CORE_MATRIX_NAIVE_PURE_OVERRIDE_DECL
    ADELIE_CORE_MATRIX_NAIVE_OVERRIDE_DECL
};

} // namespace matrix
} // namespace adelie_core