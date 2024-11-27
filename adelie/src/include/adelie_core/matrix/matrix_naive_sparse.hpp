#pragma once
#include <adelie_core/matrix/matrix_naive_base.hpp>

#ifndef ADELIE_CORE_MATRIX_NAIVE_SPARSE_TP
#define ADELIE_CORE_MATRIX_NAIVE_SPARSE_TP \
    template <class SparseType, class IndexType>
#endif
#ifndef ADELIE_CORE_MATRIX_NAIVE_SPARSE
#define ADELIE_CORE_MATRIX_NAIVE_SPARSE \
    MatrixNaiveSparse<SparseType, IndexType>
#endif

namespace adelie_core {
namespace matrix {

template <class SparseType, class IndexType=Eigen::Index>
class MatrixNaiveSparse: public MatrixNaiveBase<typename SparseType::Scalar, IndexType>
{
public:
    using base_t = MatrixNaiveBase<typename SparseType::Scalar, IndexType>;
    using sparse_t = SparseType;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::colmat_value_t;
    using typename base_t::rowmat_value_t;
    using typename base_t::sp_mat_value_t;
    using vec_sp_value_t = vec_value_t;
    using vec_sp_index_t = util::rowvec_type<typename sparse_t::StorageIndex>;

    static_assert(!sparse_t::IsRowMajor, "MatrixNaiveSparse: only column-major allowed!");
    
private:
    const Eigen::Map<const sparse_t> _mat;  // underlying sparse matrix
    const size_t _n_threads;                // number of threads
    vec_value_t _buff;

    inline value_t _cmul(
        int j, 
        const Eigen::Ref<const vec_value_t>& v,
        const Eigen::Ref<const vec_value_t>& weights,
        size_t n_threads,
        Eigen::Ref<vec_value_t> buff
    ) const;

    inline value_t _sq_cmul(
        int j, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> buff
    ) const;

    inline void _ctmul(
        int j, 
        value_t v, 
        Eigen::Ref<vec_value_t> out,
        size_t n_threads
    ) const;

public:
    explicit MatrixNaiveSparse(
        size_t rows,
        size_t cols,
        size_t nnz,
        const Eigen::Ref<const vec_sp_index_t>& outer,
        const Eigen::Ref<const vec_sp_index_t>& inner,
        const Eigen::Ref<const vec_sp_value_t>& value,
        size_t n_threads
    );

    ADELIE_CORE_MATRIX_NAIVE_PURE_OVERRIDE_DECL
};

} // namespace matrix
} // namespace adelie_core