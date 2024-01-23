#pragma once
#include <cstdio>
#include <adelie_core/matrix/matrix_naive_base.hpp>
#include <adelie_core/matrix/utils.hpp>

namespace adelie_core {
namespace matrix {

template <class DenseType> 
class MatrixNaiveKroneckerEye: public MatrixNaiveBase<typename DenseType::Scalar>
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
    const size_t _K;
    const size_t _n_threads;                // number of threads
    util::rowmat_type<value_t> _buff;
    
public:
    MatrixNaiveKroneckerEye(
        const Eigen::Ref<const dense_t>& mat,
        size_t K,
        size_t n_threads
    ): 
        _mat(mat.data(), mat.rows(), mat.cols()),
        _K(K),
        _n_threads(n_threads),
        _buff(_n_threads, std::min(mat.rows(), mat.cols()))
    {}

    
    value_t cmul(
        int j, 
        const Eigen::Ref<const vec_value_t>& v
    ) override 
    {
        base_t::check_cmul(j, v.size(), rows(), cols());
        Eigen::Map<const rowmat_value_t> V(v.data(), rows() / _K, _K);
        int i = j / _K;
        int l = j - _K * i;
        Eigen::Map<vec_value_t> _vbuff(_buff.data(), _n_threads);
        return ddot(_mat.col(i), V.col(l), _n_threads, _vbuff);
    }

    void ctmul(
        int j, 
        value_t v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) const override
    {
        base_t::check_ctmul(j, weights.size(), out.size(), rows(), cols());
        Eigen::Map<rowmat_value_t> Out(out.data(), rows() / _K, _K);
        Eigen::Map<const rowmat_value_t> W(weights.data(), Out.rows(), Out.cols());
        int i = j / _K;
        int l = j - _K * i;
        dvzero(out, _n_threads);
        Out.col(l) = v * _mat.col(i).cwiseProduct(W.col(l));
    }

    void bmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {
    }

    void btmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override
    {
    }

    void mul(
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {

    }

    void cov(
        int j, int q,
        const Eigen::Ref<const vec_value_t>& sqrt_weights,
        Eigen::Ref<colmat_value_t> out,
        Eigen::Ref<colmat_value_t> buffer
    ) const override
    {

    }

    int rows() const override { return _K * _mat.rows(); }
    int cols() const override { return _K * _mat.cols(); }

    /* Non-speed critical routines */

    void sp_btmul(
        const sp_mat_value_t& v,
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<rowmat_value_t> out
    ) const override
    {

    }

    void means(
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) const override
    {

    }
};

} // namespace matrix
} // namespace adelie_core