#pragma once
#include <adelie_core/matrix/matrix_base.hpp>

namespace adelie_core {
namespace matrix {
    
template <class DenseType>
class MatrixDense: public MatrixBase<typename DenseType::Scalar>
{
public:
    using base_t = MatrixBase<typename DenseType::Scalar>;
    using dense_t = DenseType;
    using typename base_t::value_t;
    using typename base_t::rowvec_t;
    
private:
    const Eigen::Map<const dense_t> _mat;
    
public:
    MatrixDense(const Eigen::Ref<const dense_t>& mat)
        : _mat(mat.data(), mat.rows(), mat.cols())
    {}
    
    value_t cmul(
        int j, 
        const Eigen::Ref<const rowvec_t>& v
    ) const override
    {
        return _mat.col(j).dot(v.matrix());
    }

    void ctmul(
        int j, 
        value_t v, 
        Eigen::Ref<rowvec_t> out
    ) const override
    {
        out.matrix() = v * _mat.col(j);
    }

    void bmul(
        int i, int j, int p, int q, 
        const Eigen::Ref<const rowvec_t>& v, 
        Eigen::Ref<rowvec_t> out
    ) const override
    {
        out.matrix().noalias() = v.matrix() * _mat.block(i, j, p, q);
    }

    void btmul(
        int i, int j, int p, int q, 
        const Eigen::Ref<const rowvec_t>& v, 
        Eigen::Ref<rowvec_t> out
    ) const override
    {
        out.matrix().noalias() = v.matrix() * _mat.block(i, j, p, q).transpose();
    }

    value_t cnormsq(int j) const override
    {
        return _mat.col(j).squaredNorm();
    }

    int rows() const override
    {
        return _mat.rows();
    }
    
    int cols() const override
    {
        return _mat.cols();
    }
};

} // namespace matrix
} // namespace adelie_core