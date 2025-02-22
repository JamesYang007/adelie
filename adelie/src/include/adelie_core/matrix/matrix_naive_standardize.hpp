#pragma once
#include <adelie_core/matrix/matrix_naive_base.hpp>

#ifndef ADELIE_CORE_MATRIX_NAIVE_STANDARDIZE_TP
#define ADELIE_CORE_MATRIX_NAIVE_STANDARDIZE_TP \
    template <class ValueType, class IndexType>
#endif
#ifndef ADELIE_CORE_MATRIX_NAIVE_STANDARDIZE
#define ADELIE_CORE_MATRIX_NAIVE_STANDARDIZE \
    MatrixNaiveStandardize<ValueType, IndexType>
#endif

namespace adelie_core {
namespace matrix {

template <class ValueType, class IndexType=Eigen::Index>
class MatrixNaiveStandardize: public MatrixNaiveBase<ValueType, IndexType>
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
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;

private:
    base_t* _mat;
    const map_cvec_value_t _centers;
    const map_cvec_value_t _scales;
    const size_t _n_threads;
    vec_value_t _buff;

public:
    explicit MatrixNaiveStandardize(
        base_t& mat,
        const Eigen::Ref<const vec_value_t>& centers,
        const Eigen::Ref<const vec_value_t>& scales,
        size_t n_threads
    );

    ADELIE_CORE_MATRIX_NAIVE_PURE_OVERRIDE_DECL
    ADELIE_CORE_MATRIX_NAIVE_OVERRIDE_DECL
};

} // namespace matrix
} // namespace adelie_core 