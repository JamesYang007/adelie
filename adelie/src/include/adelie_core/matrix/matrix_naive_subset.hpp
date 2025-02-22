#pragma once
#include <adelie_core/matrix/matrix_naive_base.hpp>

#ifndef ADELIE_CORE_MATRIX_NAIVE_CSUBSET_TP
#define ADELIE_CORE_MATRIX_NAIVE_CSUBSET_TP \
    template <class ValueType, class IndexType>
#endif
#ifndef ADELIE_CORE_MATRIX_NAIVE_CSUBSET
#define ADELIE_CORE_MATRIX_NAIVE_CSUBSET \
    MatrixNaiveCSubset<ValueType, IndexType>
#endif

#ifndef ADELIE_CORE_MATRIX_NAIVE_RSUBSET_TP
#define ADELIE_CORE_MATRIX_NAIVE_RSUBSET_TP \
    template <class ValueType, class IndexType>
#endif
#ifndef ADELIE_CORE_MATRIX_NAIVE_RSUBSET
#define ADELIE_CORE_MATRIX_NAIVE_RSUBSET \
    MatrixNaiveRSubset<ValueType, IndexType>
#endif

namespace adelie_core {
namespace matrix {

template <class ValueType, class IndexType=Eigen::Index>
class MatrixNaiveCSubset: public MatrixNaiveBase<ValueType, IndexType>
{
public:
    using base_t = MatrixNaiveBase<ValueType, IndexType>;
    using typename base_t::value_t;
    using typename base_t::index_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::colmat_value_t;
    using typename base_t::rowmat_value_t;
    using typename base_t::sp_mat_value_t;
    using dyn_vec_index_t = std::vector<index_t>;
    using map_cvec_index_t = Eigen::Map<const vec_index_t>;
    
private:
    base_t* _mat;               // underlying matrix
    const map_cvec_index_t _subset;  // column subset
    const std::tuple<
        vec_index_t,
        dyn_vec_index_t
    > _subset_cinfo;            // 1) number of elements left
                                // in the contiguous sub-chunk
                                // starting at _subset[i]
                                // 2) beginning index to each
                                // contiguous sub-chunk
    const size_t _n_threads;

    static inline auto init_subset_cinfo(
        const Eigen::Ref<const vec_index_t>& subset
    );

public:
    explicit MatrixNaiveCSubset(
        base_t& mat,
        const Eigen::Ref<const vec_index_t>& subset,
        size_t n_threads
    );

    ADELIE_CORE_MATRIX_NAIVE_PURE_OVERRIDE_DECL
    ADELIE_CORE_MATRIX_NAIVE_OVERRIDE_DECL
};

template <class ValueType, class IndexType=Eigen::Index>
class MatrixNaiveRSubset: public MatrixNaiveBase<ValueType, IndexType>
{
public:
    using base_t = MatrixNaiveBase<ValueType, IndexType>;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::colmat_value_t;
    using typename base_t::rowmat_value_t;
    using typename base_t::sp_mat_value_t;
    using vec_bool_t = util::rowvec_type<bool>;
    using map_cvec_index_t = Eigen::Map<const vec_index_t>;
    
private:
    base_t* _mat;               // underlying matrix
    const map_cvec_index_t _subset;
    const vec_value_t _mask;  
    const size_t _n_threads;
    vec_value_t _buffer;

    static inline auto init_mask(
        size_t n,
        const Eigen::Ref<const vec_index_t>& subset
    );
    
public:
    explicit MatrixNaiveRSubset(
        base_t& mat,
        const Eigen::Ref<const vec_index_t>& subset,
        size_t n_threads
    );

    ADELIE_CORE_MATRIX_NAIVE_PURE_OVERRIDE_DECL
    ADELIE_CORE_MATRIX_NAIVE_OVERRIDE_DECL
};

} // namespace matrix
} // namespace adelie_core