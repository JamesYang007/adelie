#pragma once
#include <adelie_core/matrix/matrix_cov_base.hpp>
#include <adelie_core/matrix/utils.hpp>

namespace adelie_core {
namespace matrix {

template <class DenseType>
class MatrixCovDense: public MatrixCovBase<typename DenseType::Scalar>
{
public:
    using base_t = MatrixCovBase<typename DenseType::Scalar>;
    using dense_t = DenseType;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using typename base_t::colmat_value_t;
    
private:
    const Eigen::Map<const dense_t> _mat;   // underlying dense matrix
    const size_t _n_threads;                // number of threads
    util::rowmat_type<value_t> _buff;
    
public:
    explicit MatrixCovDense(
        const Eigen::Ref<const dense_t>& mat,
        size_t n_threads
    ): 
        _mat(mat.data(), mat.rows(), mat.cols()),
        _n_threads(n_threads),
        _buff(_n_threads, std::min(mat.rows(), mat.cols()))
    {}

    using base_t::rows;
    
    void bmul(
        int i, int j, int p, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_bmul(i, j, p, q, v.size(), out.size(), rows(), cols());
        out.matrix().noalias() = v.matrix() * _mat.block(i, j, p, q);
    }

    void mul(
        int i, int p,
        const Eigen::Ref<const vec_value_t>& v,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_mul(i, p, v.size(), out.size(), rows(), cols());
        auto outm = out.matrix();
        dgemv(
            _mat.middleCols(i, p).transpose(),
            v.matrix(),
            _n_threads,
            _buff,
            outm
        );
    } 

    void to_dense(
        int i, int p,
        Eigen::Ref<colmat_value_t> out
    ) override
    {
        base_t::check_to_dense(i, p, out.rows(), out.cols(), rows(), cols());
        out = _mat.block(i, i, p, p);
    }

    int cols() const override
    {
        return _mat.cols();
    }
};

} // namespace matrix
} // namespace adelie_core