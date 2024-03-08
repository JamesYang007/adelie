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
        _buff(3 * mat.rows() + mat.cols())
    {}

    value_t cmul(
        int j, 
        const Eigen::Ref<const vec_value_t>& v,
        const Eigen::Ref<const vec_value_t>& weights
    ) override 
    {
        base_t::check_cmul(j, v.size(), weights.size(), rows(), cols());
        Eigen::Map<const rowmat_value_t> V(v.data(), rows() / _K, _K);
        Eigen::Map<const rowmat_value_t> W(weights.data(), V.rows(), V.cols());
        int i = j / _K;
        int l = j - _K * i;
        Eigen::Map<vec_value_t> _v(_buff.data(), V.rows());
        Eigen::Map<vec_value_t> _w(_buff.data() + V.rows(), V.rows());
        _v = V.col(l);
        _w = W.col(l);
        return _mat->cmul(i, _v, _w);
    }

    void ctmul(
        int j, 
        value_t v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_ctmul(j, out.size(), rows(), cols());
        Eigen::Map<rowmat_value_t> Out(out.data(), rows() / _K, _K);
        int i = j / _K;
        int l = j - _K * i;
        dvzero(out, _n_threads);
        Eigen::Map<vec_value_t> _out(_buff.data(), Out.rows());
        _mat->ctmul(i, v, _out);
        Out.col(l) = _out;
    }

    void bmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_bmul(j, q, v.size(), weights.size(), out.size(), rows(), cols());
        Eigen::Map<const rowmat_value_t> V(v.data(), rows() / _K, _K);
        Eigen::Map<const rowmat_value_t> W(weights.data(), V.rows(), V.cols());
        Eigen::Map<vec_value_t> _v(_buff.data(), V.rows());
        Eigen::Map<vec_value_t> _w(_buff.data() + V.rows(), V.rows());
        for (int l = 0; l < _K; ++l) {
            const auto j_l = std::max(j-l, 0);
            const auto i_begin = j_l / static_cast<int>(_K) + ((j_l % _K) != 0);
            if (j-l+q <= 0) continue;
            const auto i_end = (j-l+q-1) / static_cast<int>(_K) + 1;
            const auto i_q = i_end - i_begin;
            if (i_q <= 0) continue;
            _v = V.col(l);
            _w = W.col(l);
            Eigen::Map<vec_value_t> _out(
                _buff.data() + 2 * V.rows(),
                i_q
            );
            _mat->bmul(i_begin, i_q, _v, _w, _out);
            for (int i = i_begin; i < i_end; ++i){
                out[i*_K+l-j] = _out[i-i_begin];
            }
        }
    }

    void btmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_btmul(j, q, v.size(), out.size(), rows(), cols());
        Eigen::Map<rowmat_value_t> Out(out.data(), rows() / _K, _K);
        for (int l = 0; l < _K; ++l) {
            const auto j_l = std::max(j-l, 0);
            const auto i_begin = j_l / static_cast<int>(_K) + ((j_l % _K) != 0);
            if (j-l+q <= 0) {
                Out.col(l).setZero();
                continue;
            }
            const auto i_end = (j-l+q-1) / static_cast<int>(_K) + 1;
            const auto i_q = i_end - i_begin;
            if (i_q <= 0) {
                Out.col(l).setZero();
                continue;
            }
            Eigen::Map<vec_value_t> _v(_buff.data(), i_q);
            for (int i = i_begin; i < i_end; ++i) {
                _v[i-i_begin] = v[i*_K+l-j];
            }
            Eigen::Map<vec_value_t> _out(
                _buff.data() + i_q,
                Out.rows()
            );
            _mat->btmul(i_begin, i_q, _v, _out);
            Out.col(l) = _out;
        }
    }

    void mul(
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        bmul(0, cols(), v, weights, out);
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
            if (j-l+q <= 0) continue;
            const auto i_end = (j-l+q-1) / static_cast<int>(_K) + 1;
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
        Eigen::Ref<rowmat_value_t> out
    ) override
    {
        base_t::check_sp_btmul(
            v.rows(), v.cols(), out.rows(), out.cols(), rows(), cols()
        );

        std::vector<int> outers(v.outerSize()+1); 
        outers[0] = 0;
        std::vector<int> inners;
        std::vector<value_t> values;
        inners.reserve(v.nonZeros());
        values.reserve(v.nonZeros());

        rowmat_value_t _out(out.rows(), rows() / _K);

        for (int l = 0; l < _K; ++l) {
            inners.clear();
            values.clear();

            // populate current v
            for (int k = 0; k < v.outerSize(); ++k) {
                typename sp_mat_value_t::InnerIterator it(v, k);
                int n_read = 0;
                for (; it; ++it) {
                    if (it.index() % _K != l) continue;
                    inners.push_back(it.index() / _K);
                    values.push_back(it.value());
                    ++n_read;
                }
                outers[k+1] = outers[k] + n_read;
            }
            Eigen::Map<sp_mat_value_t> _v(
                v.rows(),
                v.cols() / _K,
                inners.size(),
                outers.data(),
                inners.data(),
                values.data()
            );

            _mat->sp_btmul(_v, _out);

            #pragma omp parallel for schedule(static) num_threads(_n_threads)
            for (int k = 0; k < out.rows(); ++k) {
                Eigen::Map<rowmat_value_t> out_k(
                    out.row(k).data(), _out.cols(), _K                
                );
                out_k.col(l) = _out.row(k);
            }
        }
    }
};

template <class DenseType> 
class MatrixNaiveKroneckerEyeDense: public MatrixNaiveBase<typename DenseType::Scalar>
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
    const Eigen::Map<const dense_t> _mat;
    const size_t _K;
    const size_t _n_threads;
    rowmat_value_t _buff;
    vec_value_t _vbuff;
    
public:
    explicit MatrixNaiveKroneckerEyeDense(
        const Eigen::Ref<const dense_t>& mat,
        size_t K,
        size_t n_threads
    ): 
        _mat(mat.data(), mat.rows(), mat.cols()),
        _K(K),
        _n_threads(n_threads),
        _buff(_n_threads, K),
        _vbuff(mat.rows() * K)
    {}

    value_t cmul(
        int j, 
        const Eigen::Ref<const vec_value_t>& v,
        const Eigen::Ref<const vec_value_t>& weights
    ) override 
    {
        base_t::check_cmul(j, v.size(), weights.size(), rows(), cols());
        Eigen::Map<const rowmat_value_t> V(v.data(), rows() / _K, _K);
        Eigen::Map<const rowmat_value_t> W(weights.data(), V.rows(), V.cols());
        const int i = j / _K;
        const int l = j - _K * i;
        Eigen::Map<vec_value_t> vbuff(_buff.data(), _n_threads);
        return ddot(V.col(l).cwiseProduct(W.col(l)), _mat.col(i), _n_threads, vbuff);
    }

    void ctmul(
        int j, 
        value_t v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_ctmul(j, out.size(), rows(), cols());
        Eigen::Map<rowmat_value_t> Out(out.data(), rows() / _K, _K);
        const int i = j / _K;
        const int l = j - _K * i;
        dvzero(out, _n_threads);
        auto _out = Out.col(l);
        dax(v, _mat.col(i), _n_threads, _out);
    }

    void bmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_bmul(j, q, v.size(), weights.size(), out.size(), rows(), cols());
        dvveq(_vbuff, v * weights, _n_threads);
        Eigen::Map<const rowmat_value_t> VW(_vbuff.data(), rows() / _K, _K);
        int n_processed = 0;
        while (n_processed < q) {
            const int i = (j + n_processed) / _K;
            const int l = (j + n_processed) - _K * i;
            const int size = std::min<int>(_K-l, q-n_processed);
            auto _out = out.segment(n_processed, size).matrix();
            dgemv(
                VW.middleCols(l, size), 
                _mat.col(i).transpose(), 
                _n_threads, 
                _buff, 
                _out
            );
            n_processed += size;
        }
    }

    void btmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_btmul(j, q, v.size(), out.size(), rows(), cols());
        Eigen::Map<rowmat_value_t> Out(out.data(), rows() / _K, _K);
        dvzero(out, _n_threads);
        int n_processed = 0;
        while (n_processed < q) {
            const int i = (j + n_processed) / _K;
            const int l = (j + n_processed) - _K * i;
            const int size = std::min<int>(_K-l, q-n_processed);
            Eigen::Map<const vec_value_t> _v(v.data() + n_processed, size);
            for (int k = 0; k < size; ++k) {
                auto _out = Out.col(l+k);
                dvaddi(_out, _v[k] * _mat.col(i), _n_threads);
            }
            n_processed += size;
        }
    }

    void mul(
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_bmul(0, cols(), v.size(), weights.size(), out.size(), rows(), cols());
        dvveq(_vbuff, v * weights, _n_threads);
        Eigen::Map<const rowmat_value_t> VW(_vbuff.data(), rows() / _K, _K);
        Eigen::Map<rowmat_value_t> Out(out.data(), cols() / _K, _K);
        Eigen::setNbThreads(_n_threads);
        Out.noalias() = _mat.transpose() * VW;
        Eigen::setNbThreads(1);
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
            if (j-l+q <= 0) continue;
            const auto i_end = (j-l+q-1) / static_cast<int>(_K) + 1;
            const auto i_q = i_end - i_begin;
            if (i_q <= 0) continue;

            auto buff = buffer.topLeftCorner(_mat.rows(), i_q);
            auto buff_array = buff.array();
            dmmeq(
                buff_array,
                _mat.middleCols(i_begin, i_q).array().colwise() * sqrt_W.col(l).array(),
                _n_threads
            );
            for (int i1 = 0; i1 < i_q; ++i1) {
                for (int i2 = 0; i2 < i_q; ++i2) {
                    out((i1+i_begin)*_K+l-j, (i2+i_begin)*_K+l-j) = (
                        buff.col(i1).dot(buff.col(i2))
                    );
                }
            }
        }
    }

    int rows() const override { return _K * _mat.rows(); }
    int cols() const override { return _K * _mat.cols(); }

    void sp_btmul(
        const sp_mat_value_t& v,
        Eigen::Ref<rowmat_value_t> out
    ) override
    {
        base_t::check_sp_btmul(
            v.rows(), v.cols(), out.rows(), out.cols(), rows(), cols()
        );

        std::vector<int> outers(v.outerSize()+1); 
        outers[0] = 0;
        std::vector<int> inners;
        std::vector<value_t> values;
        inners.reserve(v.nonZeros());
        values.reserve(v.nonZeros());

        rowmat_value_t _out(out.rows(), rows() / _K);

        for (int l = 0; l < _K; ++l) {
            inners.clear();
            values.clear();

            // populate current v
            for (int k = 0; k < v.outerSize(); ++k) {
                typename sp_mat_value_t::InnerIterator it(v, k);
                int n_read = 0;
                for (; it; ++it) {
                    if (it.index() % _K != l) continue;
                    inners.push_back(it.index() / _K);
                    values.push_back(it.value());
                    ++n_read;
                }
                outers[k+1] = outers[k] + n_read;
            }
            Eigen::Map<sp_mat_value_t> _v(
                v.rows(),
                v.cols() / _K,
                inners.size(),
                outers.data(),
                inners.data(),
                values.data()
            );

            _out.noalias() = _v * _mat.transpose();

            #pragma omp parallel for schedule(static) num_threads(_n_threads)
            for (int k = 0; k < out.rows(); ++k) {
                Eigen::Map<rowmat_value_t> out_k(
                    out.row(k).data(), _out.cols(), _K
                );
                out_k.col(l) = _out.row(k);
            }
        }
    }
};

} // namespace matrix
} // namespace adelie_core