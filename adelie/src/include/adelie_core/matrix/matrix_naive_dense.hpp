#pragma once
#include <adelie_core/matrix/matrix_naive_base.hpp>
#include <adelie_core/matrix/utils.hpp>

namespace adelie_core {
namespace matrix {

template <class DenseType>
class MatrixNaiveDense: public MatrixNaiveBase<typename DenseType::Scalar>
{
public:
    using base_t = MatrixNaiveBase<typename DenseType::Scalar>;
    using dense_t = DenseType;
    using typename base_t::value_t;
    using typename base_t::rowvec_t;
    using typename base_t::colmat_t;
    
private:
    const Eigen::Map<const dense_t> _mat;   // underlying dense matrix
    const size_t _n_threads;                // number of threads
    util::rowmat_type<value_t> _buff;
    
public:
    MatrixNaiveDense(
        const Eigen::Ref<const dense_t>& mat,
        size_t n_threads
    ): 
        _mat(mat.data(), mat.rows(), mat.cols()),
        _n_threads(n_threads),
        _buff(_n_threads, std::min(mat.rows(), mat.cols()))
    {}
    
    value_t cmul(
        int j, 
        const Eigen::Ref<const rowvec_t>& v
    ) const override
    {
        return ddot(_mat.col(j).matrix(), v.matrix(), _n_threads);
    }

    void ctmul(
        int j, 
        value_t v, 
        Eigen::Ref<rowvec_t> out
    ) const override
    {
        dax(v, _mat.col(j), _n_threads, out);
    }

    void bmul(
        int j, int q, 
        const Eigen::Ref<const rowvec_t>& v, 
        Eigen::Ref<rowvec_t> out
    ) override
    {
        auto outm = out.matrix();
        dgemv(
            _mat.middleCols(j, q),
            v.matrix(),
            _n_threads,
            _buff,
            outm
        );
    }

    void btmul(
        int j, int q, 
        const Eigen::Ref<const rowvec_t>& v, 
        Eigen::Ref<rowvec_t> out
    ) override
    {
        auto outm = out.matrix();
        dgemv(
            _mat.middleCols(j, q).transpose(),
            v.matrix(),
            _n_threads,
            _buff,
            outm
        );
    }

    void to_dense(
        int j, int q,
        Eigen::Ref<colmat_t> out
    ) const override
    {
        out = _mat.middleCols(j, q);
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