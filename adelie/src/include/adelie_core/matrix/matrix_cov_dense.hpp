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
    MatrixCovDense(
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
        auto outm = out.matrix();
        dgemv(
            _mat.block(i, j, p, q),
            v.matrix(),
            _n_threads,
            _buff,
            outm
        );
    }

    void to_dense(
        int i, int j, int p, int q,
        Eigen::Ref<colmat_value_t> out
    ) const override
    {
        base_t::check_to_dense(i, j, p, q, out.rows(), out.cols(), rows(), cols());
        out = _mat.block(i, j, p, q);
    }

    int cols() const override
    {
        return _mat.cols();
    }
};

} // namespace matrix
} // namespace adelie_core