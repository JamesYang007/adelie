#pragma once
#include <cstdio>
#include <adelie_core/matrix/matrix_naive_base.hpp>
#include <adelie_core/matrix/utils.hpp>

namespace adelie_core {
namespace matrix {

template <class ValueType> 
class MatrixNaiveKroneckerEye: public MatrixNaiveBase<ValueType>
{
public:
    using base_t = MatrixNaiveBase<ValueType>;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::colmat_value_t;
    using typename base_t::rowmat_value_t;
    using typename base_t::sp_mat_value_t;
    
private:
    base_t* _mat;
    const size_t _K;
    const size_t _n_threads;
    vec_value_t _buff;
    
public:
    MatrixNaiveKroneckerEye(
        base_t& mat,
        size_t K,
        size_t n_threads
    ): 
        _mat(&mat),
        _K(K),
        _n_threads(n_threads),
        _buff(2 * mat.rows() + mat.cols())
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
        Eigen::Map<vec_value_t> _vbuff(
            _buff.data(),
            V.rows()
        );
        _vbuff = V.col(l);
        return _mat->cmul(i, _vbuff);
    }

    void ctmul(
        int j, 
        value_t v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_ctmul(j, weights.size(), out.size(), rows(), cols());
        Eigen::Map<rowmat_value_t> Out(out.data(), rows() / _K, _K);
        Eigen::Map<const rowmat_value_t> W(weights.data(), Out.rows(), Out.cols());
        int i = j / _K;
        int l = j - _K * i;
        dvzero(out, _n_threads);
        Eigen::Map<vec_value_t> _weights(
            _buff.data(),
            W.rows()
        );
        _weights = W.col(l);
        Eigen::Map<vec_value_t> _out(
            _buff.data() + W.rows(),
            W.rows()
        );
        _mat->ctmul(i, v, _weights, _out);
        Out.col(l) = _out;
    }

    void bmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_bmul(j, q, v.size(), out.size(), rows(), cols());
        Eigen::Map<const rowmat_value_t> V(v.data(), rows() / _K, _K);
        Eigen::Map<vec_value_t> _v(_buff.data(), V.rows());
        for (int l = 0; l < _K; ++l) {
            const auto j_l = std::max(j-l, 0);
            const auto i_begin = j_l / static_cast<int>(_K) + ((j_l % _K) != 0);
            const auto i_end = std::max(j-l+q-1, 0) / static_cast<int>(_K) + 1;
            const auto i_q = i_end - i_begin;
            if (i_q <= 0) continue;
            _v = V.col(l);
            Eigen::Map<vec_value_t> _out(
                _buff.data() + V.rows(),
                i_q
            );
            _mat->bmul(i_begin, i_q, _v, _out);
            for (int i = i_begin; i < i_end; ++i){
                out[i*_K+l] = _out[i-i_begin];
            }
        }
    }

    void btmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_btmul(j, q, v.size(), weights.size(), out.size(), rows(), cols());
        Eigen::Map<const rowmat_value_t> W(weights.data(), rows() / _K, _K);
        Eigen::Map<vec_value_t> _weights(_buff.data(), W.rows());
        Eigen::Map<rowmat_value_t> Out(out.data(), W.rows(), W.cols());
        for (int l = 0; l < _K; ++l) {
            const auto j_l = std::max(j-l, 0);
            const auto i_begin = j_l / static_cast<int>(_K) + ((j_l % _K) != 0);
            const auto i_end = std::max(j-l+q-1, 0) / static_cast<int>(_K) + 1;
            const auto i_q = i_end - i_begin;
            if (i_q <= 0) {
                Out.col(l).setZero();
                continue;
            }
            _weights = W.col(l);
            Eigen::Map<vec_value_t> _v(
                _buff.data() + _weights.size(),
                i_q
            );
            for (int i = i_begin; i < i_end; ++i) {
                _v[i-i_begin] = v[i*_K+l];
            }
            Eigen::Map<vec_value_t> _out(
                _buff.data() + _weights.size() + i_q,
                _weights.size()
            );
            _mat->btmul(i_begin, i_q, _v, _weights, _out);
            Out.col(l) = _out;
        }
    }

    void mul(
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {
        bmul(0, cols(), v, out);
    }

    void cov(
        int j, int q,
        const Eigen::Ref<const vec_value_t>& sqrt_weights,
        Eigen::Ref<colmat_value_t> out,
        Eigen::Ref<colmat_value_t> buffer
    ) override
    {
        base_t::check_cov(
            j, q, sqrt_weights.size(), 
            out.rows(), out.cols(), buffer.rows(), buffer.cols(), 
            rows(), cols()
        );
        Eigen::Map<const rowmat_value_t> sqrt_W(sqrt_weights.data(), rows() / _K, _K);
        out.setZero(); // do NOT parallelize!
        for (int l = 0; l < _K; ++l) {
            const auto j_l = std::max(j-l, 0);
            const auto i_begin = j_l / static_cast<int>(_K) + ((j_l % _K) != 0);
            const auto i_end = std::max(j-l+q-1, 0) / static_cast<int>(_K) + 1;
            const auto i_q = i_end - i_begin;
            if (i_q <= 0) continue;
            if (_buff.size() < sqrt_W.rows() + i_q * i_q) {
                _buff.resize(_buff.size() + i_q * i_q);
            }
            Eigen::Map<vec_value_t> _sqrt_weights(_buff.data(), sqrt_W.rows());
            _sqrt_weights = sqrt_W.col(l);
            Eigen::Map<colmat_value_t> _out(
                _buff.data() + _sqrt_weights.size(),
                i_q, i_q
            );
            Eigen::Map<colmat_value_t> _buffer(
                buffer.data(),
                _mat->rows(),
                i_q
            );
            _mat->cov(i_begin, i_q, _sqrt_weights, _out, _buffer);
            for (int i1 = 0; i1 < i_q; ++i1) {
                for (int i2 = 0; i2 < i_q; ++i2) {
                    out((i1+i_begin)*_K+l-j, (i2+i_begin)*_K+l-j) = _out(i1, i2);
                }
            }
        }
    }

    int rows() const override { return _K * _mat->rows(); }
    int cols() const override { return _K * _mat->cols(); }

    /* Non-speed critical routines */

    void sp_btmul(
        const sp_mat_value_t& v,
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<rowmat_value_t> out
    ) override
    {

    }
};

} // namespace matrix
} // namespace adelie_core