#pragma once
#include <adelie_core/matrix/matrix_naive_base.hpp>

#ifndef ADELIE_CORE_MATRIX_NAIVE_KRONECKER_EYE_TP
#define ADELIE_CORE_MATRIX_NAIVE_KRONECKER_EYE_TP \
    template <class ValueType, class IndexType>
#endif
#ifndef ADELIE_CORE_MATRIX_NAIVE_KRONECKER_EYE
#define ADELIE_CORE_MATRIX_NAIVE_KRONECKER_EYE \
    MatrixNaiveKroneckerEye<ValueType, IndexType>
#endif

#ifndef ADELIE_CORE_MATRIX_NAIVE_KRONECKER_EYE_DENSE_TP
#define ADELIE_CORE_MATRIX_NAIVE_KRONECKER_EYE_DENSE_TP \
    template <class DenseType, class IndexType>
#endif
#ifndef ADELIE_CORE_MATRIX_NAIVE_KRONECKER_EYE_DENSE
#define ADELIE_CORE_MATRIX_NAIVE_KRONECKER_EYE_DENSE \
    MatrixNaiveKroneckerEyeDense<DenseType, IndexType>
#endif

namespace adelie_core {
namespace matrix {

template <class ValueType, class IndexType=Eigen::Index> 
class MatrixNaiveKroneckerEye: public MatrixNaiveBase<ValueType, IndexType>
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
    base_t* _mat;
    const size_t _K;
    const size_t _n_threads;
    vec_value_t _buff;
    
public:
    explicit MatrixNaiveKroneckerEye(
        base_t& mat,
        size_t K,
        size_t n_threads
    );

    ADELIE_CORE_MATRIX_NAIVE_PURE_OVERRIDE_DECL

    virtual void mean(
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override;

    virtual void var(
        const Eigen::Ref<const vec_value_t>& centers,
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override;
};

template <class DenseType, class IndexType=Eigen::Index> 
class MatrixNaiveKroneckerEyeDense: public MatrixNaiveBase<typename DenseType::Scalar, IndexType>
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
    const Eigen::Map<const dense_t> _mat;
    const size_t _K;
    const size_t _n_threads;
    rowmat_value_t _buff;
    vec_value_t _vbuff;
    
public:
    explicit MatrixNaiveKroneckerEyeDense(
        const Eigen::Ref<const dense_t>& mat,
        size_t K,
        size_t n_threads
    );

    ADELIE_CORE_MATRIX_NAIVE_PURE_OVERRIDE_DECL

    virtual void mean(
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override;

    virtual void var(
        const Eigen::Ref<const vec_value_t>& centers,
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override;
};

} // namespace matrix
} // namespace adelie_core