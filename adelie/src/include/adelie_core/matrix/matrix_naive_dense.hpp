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
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::colmat_value_t;
    using typename base_t::rowmat_value_t;
    using typename base_t::sp_mat_value_t;
    
private:
    const Eigen::Map<const dense_t> _mat;   // underlying dense matrix
    const size_t _n_threads;                // number of threads
    util::rowmat_type<value_t> _buff;
    util::rowvec_type<value_t> _vbuff;
    
public:
    MatrixNaiveDense(
        const Eigen::Ref<const dense_t>& mat,
        size_t n_threads
    ): 
        _mat(mat.data(), mat.rows(), mat.cols()),
        _n_threads(n_threads),
        _buff(_n_threads, std::min(mat.rows(), mat.cols())),
        _vbuff(mat.rows())
    {}
    
    value_t cmul(
        int j, 
        const Eigen::Ref<const vec_value_t>& v
    ) const override
    {
        return ddot(_mat.col(j), v.matrix(), _n_threads);
    }

    void ctmul(
        int j, 
        value_t v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) const override
    {
        dax(v, _mat.transpose().row(j).array() * weights, _n_threads, out);
    }

    void bmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
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
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
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
        out *= weights;
    }

    void mul(
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {
        auto outm = out.matrix();
        dgemv(
            _mat,
            v.matrix(),
            _n_threads,
            _buff,
            outm
        );
    }

    int rows() const override
    {
        return _mat.rows();
    }
    
    int cols() const override
    {
        return _mat.cols();
    }

    void cov(
        int j, int q,
        const Eigen::Ref<const vec_value_t>& sqrt_weights,
        Eigen::Ref<colmat_value_t> out,
        Eigen::Ref<colmat_value_t> buffer
    ) const override
    {
        auto& Xj = buffer;
        
        Xj.transpose().array() = (
            _mat.middleCols(j, q).transpose().array().rowwise() * sqrt_weights
        );

        Eigen::setNbThreads(_n_threads);
        out.noalias() = Xj.transpose() * Xj;
        Eigen::setNbThreads(0);
    }

    void sp_btmul(
        int j, int q, 
        const sp_mat_value_t& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<rowmat_value_t> out
    ) const override
    {
        out.noalias() = v * _mat.middleCols(j, q).transpose();
        out.array().rowwise() *= weights;
    }

    void to_dense(
        int j, int q,
        Eigen::Ref<colmat_value_t> out
    ) const override
    {
        dmmeq(
            out,
            _mat.middleCols(j, q),
            _n_threads
        );
    }

    void means(
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) const override
    {
        const size_t p = _mat.cols();
        const int n_blocks = std::min<int>(_n_threads, p);
        const int block_size = p / n_blocks;
        const int remainder = p % n_blocks;
        #pragma omp parallel for schedule(static) num_threads(n_blocks)
        for (int t = 0; t < n_blocks; ++t) {
            const auto begin = (
                std::min<int>(t, remainder) * (block_size + 1) 
                + std::max<int>(t-remainder, 0) * block_size
            );
            const auto size = block_size + (t < remainder);
            for (int j = 0; j < size; ++j) {
                out[begin + j] = _mat.col(begin + j).dot(weights.matrix());
            }
        }
    }
};

} // namespace matrix
} // namespace adelie_core