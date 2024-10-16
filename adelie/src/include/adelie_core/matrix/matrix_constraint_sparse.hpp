#pragma once
#include <adelie_core/matrix/matrix_constraint_base.hpp>

namespace adelie_core {
namespace matrix {

template <class SparseType, class IndexType=Eigen::Index>
class MatrixConstraintSparse: public MatrixConstraintBase<typename SparseType::Scalar, IndexType>
{
public:
    using base_t = MatrixConstraintBase<typename SparseType::Scalar, IndexType>;
    using sparse_t = SparseType;
    using typename base_t::value_t;
    using typename base_t::index_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::colmat_value_t;
    using vec_sp_value_t = vec_value_t;
    using vec_sp_index_t = util::rowvec_type<typename sparse_t::StorageIndex>;

    static_assert(sparse_t::IsRowMajor, "MatrixConstraintSparse: only row-major allowed!");
    
private:
    const Eigen::Map<const sparse_t> _mat;  // underlying sparse matrix
    const size_t _n_threads;                // number of threads
    vec_value_t _buff;
    
public:
    explicit MatrixConstraintSparse(
        size_t rows,
        size_t cols,
        size_t nnz,
        const Eigen::Ref<const vec_sp_index_t>& outer,
        const Eigen::Ref<const vec_sp_index_t>& inner,
        const Eigen::Ref<const vec_sp_value_t>& value,
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