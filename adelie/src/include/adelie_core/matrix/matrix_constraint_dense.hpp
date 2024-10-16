#pragma once
#include <adelie_core/matrix/matrix_constraint_base.hpp>

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
    rowmat_value_t _buff;
    
public:
    explicit MatrixConstraintDense(
        const Eigen::Ref<const dense_t>& mat,
        size_t n_threads
    );

    void rmmul(
        int j,
        const Eigen::Ref<const colmat_value_t>& Q,
        Eigen::Ref<vec_value_t> out
    ) override;

    value_t rvmul(
        int j,
        const Eigen::Ref<const vec_value_t>& v
    ) override;

    void rvtmul(
        int j,
        value_t v,
        Eigen::Ref<vec_value_t> out
    ) override;

    void mul(
        const Eigen::Ref<const vec_value_t>& v,
        Eigen::Ref<vec_value_t> out
    ) override;

    void tmul(
        const Eigen::Ref<const vec_value_t>& v,
        Eigen::Ref<vec_value_t> out
    ) override;

    void cov(
        const Eigen::Ref<const colmat_value_t>& Q,
        Eigen::Ref<colmat_value_t> out
    ) override;

    int rows() const override;
    
    int cols() const override;

    void sp_mul(
        const Eigen::Ref<const vec_index_t>& indices,
        const Eigen::Ref<const vec_value_t>& values,
        Eigen::Ref<vec_value_t> out
    ) override;
};

} // namespace matrix
} // namespace adelie_core