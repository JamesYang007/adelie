#pragma once
#include <adelie_core/matrix/matrix_base.hpp>

namespace adelie_core {
namespace matrix {
    
template <class ValueType, int StorageOrder>
class MatrixDense: public MatrixBase<ValueType, StorageOrder>
{
    using base_t = MatrixBase<ValueType, StorageOrder>;

public:
    using typename base_t::value_t;
    using typename base_t::vec_t;
    using typename base_t::mat_t;
    
private:
    const Eigen::Map<const mat_t> _mat;
    
public:
    MatrixDense(const Eigen::Ref<const mat_t>& mat)
        : _mat(mat.data(), mat.rows(), mat.cols())
    {}
    
    Eigen::Ref<const mat_t> block(int i, int j, int p, int q) const override
    {
        return _mat.block(i, j, p, q);    
    }

    Eigen::Ref<const vec_t> col(int j) const override
    {
        return _mat.col(j);    
    }
    
    int cols() const override
    {
        return _mat.cols();
    }
};

} // namespace matrix
} // namespace adelie_core