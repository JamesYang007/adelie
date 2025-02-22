#pragma once
#include <adelie_core/matrix/matrix_naive_base.hpp>

#ifndef ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE_TP
#define ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE_TP \
    template <class ValueType, class IndexType>
#endif
#ifndef ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE
#define ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE \
    MatrixNaiveCConcatenate<ValueType, IndexType>
#endif

#ifndef ADELIE_CORE_MATRIX_NAIVE_RCONCATENATE_TP
#define ADELIE_CORE_MATRIX_NAIVE_RCONCATENATE_TP \
    template <class ValueType, class IndexType>
#endif
#ifndef ADELIE_CORE_MATRIX_NAIVE_RCONCATENATE
#define ADELIE_CORE_MATRIX_NAIVE_RCONCATENATE \
    MatrixNaiveRConcatenate<ValueType, IndexType>
#endif

namespace adelie_core {
namespace matrix {

template <class ValueType, class IndexType=Eigen::Index>
class MatrixNaiveCConcatenate: public MatrixNaiveBase<ValueType, IndexType>
{
public:
    using base_t = MatrixNaiveBase<ValueType, IndexType>;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::colmat_value_t;
    using typename base_t::rowmat_value_t;
    using typename base_t::sp_mat_value_t;
    
private:
    const std::vector<base_t*> _mat_list;   // (L,) list of naive matrices
    const size_t _rows;                     // number of rows
    const size_t _cols;                     // number of columns
    const vec_index_t _outer;               // (L+1,) outer slices for each sub-matrix.
    const vec_index_t _slice_map;           // (p,) array mapping to matrix slice
    const vec_index_t _index_map;           // (p,) array mapping to (relative) index of the slice
    const size_t _n_threads;                // number of threads

    static inline auto init_rows(
        const std::vector<base_t*>& mat_list
    );

    static inline auto init_cols(
        const std::vector<base_t*>& mat_list
    );

    static inline auto init_outer(
        const std::vector<base_t*>& mat_list
    );

    static inline auto init_slice_map(
        const std::vector<base_t*>& mat_list,
        size_t p
    );

    static inline auto init_index_map(
        const std::vector<base_t*>& mat_list,
        size_t p
    );

public:
    explicit MatrixNaiveCConcatenate(
        const std::vector<base_t*>& mat_list,
        size_t n_threads
    );

    ADELIE_CORE_MATRIX_NAIVE_PURE_OVERRIDE_DECL
    ADELIE_CORE_MATRIX_NAIVE_OVERRIDE_DECL
};

template <class ValueType, class IndexType=Eigen::Index>
class MatrixNaiveRConcatenate: public MatrixNaiveBase<ValueType, IndexType>
{
public:
    using base_t = MatrixNaiveBase<ValueType, IndexType>;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::colmat_value_t;
    using typename base_t::rowmat_value_t;
    using typename base_t::sp_mat_value_t;
    
private:
    const std::vector<base_t*> _mat_list;   // (L,) list of naive matrices
    const size_t _rows;                     // number of rows
    const size_t _cols;                     // number of columns
    const vec_index_t _outer;               // (L+1,) outer slices for each sub-matrix.
    const size_t _n_threads;                // number of threads
    vec_value_t _buff;                      // (p,) buffer

    static inline auto init_rows(
        const std::vector<base_t*>& mat_list
    );

    static inline auto init_cols(
        const std::vector<base_t*>& mat_list
    );

    static inline auto init_outer(
        const std::vector<base_t*>& mat_list
    );

public:
    explicit MatrixNaiveRConcatenate(
        const std::vector<base_t*>& mat_list,
        size_t n_threads
    );

    ADELIE_CORE_MATRIX_NAIVE_PURE_OVERRIDE_DECL
    ADELIE_CORE_MATRIX_NAIVE_OVERRIDE_DECL
};

} // namespace matrix 
} // namespace adelie_core