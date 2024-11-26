#pragma once
#include <adelie_core/matrix/matrix_constraint_base.hpp>

#ifndef ADELIE_CORE_MATRIX_CONSTRAINT_DENSE_TP
#define ADELIE_CORE_MATRIX_CONSTRAINT_DENSE_TP \
    template <class DenseType, class IndexType>
#endif
#ifndef ADELIE_CORE_MATRIX_CONSTRAINT_DENSE
#define ADELIE_CORE_MATRIX_CONSTRAINT_DENSE \
    MatrixConstraintDense<DenseType, IndexType>
#endif

namespace adelie_core {
namespace matrix {

template <class DenseType, class IndexType=Eigen::Index>
class MatrixConstraintDense: public MatrixConstraintBase<typename DenseType::Scalar, IndexType>
{
public:
    using base_t = MatrixConstraintBase<typename DenseType::Scalar, IndexType>;
    using dense_t = DenseType;
    using typename base_t::value_t;
    using typename base_t::index_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::colmat_value_t;
    using rowmat_value_t = util::rowmat_type<value_t>;
    
private:
    const Eigen::Map<const dense_t> _mat;   // underlying dense matrix
    const size_t _n_threads;                // number of threads
    
public:
    explicit MatrixConstraintDense(
        const Eigen::Ref<const dense_t>& mat,
        size_t n_threads
    );

    ADELIE_CORE_MATRIX_CONSTRAINT_PURE_OVERRIDE_DECL
};

} // namespace matrix
} // namespace adelie_core