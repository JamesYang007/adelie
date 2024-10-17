#pragma once
#include <adelie_core/matrix/matrix_naive_base.hpp>

#ifndef ADELIE_CORE_MATRIX_NAIVE_DENSE_TP
#define ADELIE_CORE_MATRIX_NAIVE_DENSE_TP \
    template <class DenseType, class IndexType>
#endif
#ifndef ADELIE_CORE_MATRIX_NAIVE_DENSE
#define ADELIE_CORE_MATRIX_NAIVE_DENSE \
    MatrixNaiveDense<DenseType, IndexType>
#endif

namespace adelie_core {
namespace matrix {

template <class DenseType, class IndexType=Eigen::Index>
class MatrixNaiveDense: public MatrixNaiveBase<typename DenseType::Scalar, IndexType>
{
public:
    using base_t = MatrixNaiveBase<typename DenseType::Scalar, IndexType>;
    using dense_t = DenseType;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::colmat_value_t;
    using typename base_t::rowmat_value_t;
    using typename base_t::sp_mat_value_t;
    
private:
    const Eigen::Map<const dense_t> _mat;   // underlying dense matrix
    const size_t _n_threads;                // number of threads
    rowmat_value_t _buff;
    vec_value_t _vbuff;
    
public:
    explicit MatrixNaiveDense(
        const Eigen::Ref<const dense_t>& mat,
        size_t n_threads
    );

    ADELIE_CORE_MATRIX_NAIVE_PURE_OVERRIDE_DECL
};

} // namespace matrix
} // namespace adelie_core