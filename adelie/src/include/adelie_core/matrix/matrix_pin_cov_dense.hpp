#pragma once
#include <adelie_core/matrix/matrix_pin_cov_base.hpp>
#include <adelie_core/matrix/utils.hpp>

namespace adelie_core {
namespace matrix {

template <class DenseType>
class MatrixPinCovDense: public MatrixPinCovBase<typename DenseType::Scalar>
{
public:
    using base_t = MatrixPinCovBase<typename DenseType::Scalar>;
    using dense_t = DenseType;
    using typename base_t::value_t;
    using typename base_t::rowvec_t;
    
private:
    const Eigen::Map<const dense_t> _mat;   // underlying dense matrix
    const size_t _n_threads;                // number of threads
    util::rowmat_type<value_t> _buff;
    
public:
    MatrixPinCovDense(
        const Eigen::Ref<const dense_t>& mat,
        size_t n_threads
    ): 
        _mat(mat.data(), mat.rows(), mat.cols()),
        _n_threads(n_threads),
        _buff(_n_threads, std::min(mat.rows(), mat.cols()))
    {}
    
    void bmul(
        int i, int j, int p, int q, 
        const Eigen::Ref<const rowvec_t>& v, 
        Eigen::Ref<rowvec_t> out
    ) override
    {
        auto outm = out.matrix();
        dgemv(
            _mat.block(i, j, p, q),
            v.matrix(),
            _n_threads,
            _buff,
            outm
        );
    }

    value_t diag(int i) const override 
    {
        return _mat.coeff(i, i);
    }

    int cols() const override
    {
        return _mat.cols();
    }
};

} // namespace matrix
} // namespace adelie_core