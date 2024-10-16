#pragma once
#include <adelie_core/matrix/matrix_cov_base.hpp>

namespace adelie_core {
namespace matrix {

template <class DenseType, class IndexType=Eigen::Index>
class MatrixCovDense: public MatrixCovBase<typename DenseType::Scalar, IndexType>
{
public:
    using base_t = MatrixCovBase<typename DenseType::Scalar, IndexType>;
    using dense_t = DenseType;
    using typename base_t::value_t;
    using typename base_t::vec_index_t;
    using typename base_t::vec_value_t;
    using typename base_t::colmat_value_t;
    
private:
    const Eigen::Map<const dense_t> _mat;   // underlying dense matrix
    const size_t _n_threads;                // number of threads
    
public:
    explicit MatrixCovDense(
        const Eigen::Ref<const dense_t>& mat,
        size_t n_threads
    ); 

    using base_t::rows;
    
    void bmul(
        const Eigen::Ref<const vec_index_t>& subset,
        const Eigen::Ref<const vec_index_t>& indices,
        const Eigen::Ref<const vec_value_t>& values,
        Eigen::Ref<vec_value_t> out
    ) override;

    void mul(
        const Eigen::Ref<const vec_index_t>& indices,
        const Eigen::Ref<const vec_value_t>& values,
        Eigen::Ref<vec_value_t> out
    ) override;

    void to_dense(
        int i, int p,
        Eigen::Ref<colmat_value_t> out
    ) override;

    int cols() const override;
};

} // namespace matrix
} // namespace adelie_core