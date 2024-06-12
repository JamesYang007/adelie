#pragma once
#include <cstdio>
#include <string>
#include <vector>
#include <adelie_core/matrix/utils.hpp>
#include <adelie_core/io/io_snp_unphased.hpp>

namespace adelie_core {
namespace matrix {

template <class ValueType,
          class MmapPtrType=std::unique_ptr<char, std::function<void(char*)>>,
          class IndexType=Eigen::Index>
class MatrixNaiveSNPUnphased: public MatrixNaiveBase<ValueType, IndexType>
{
public:
    using base_t = MatrixNaiveBase<ValueType, IndexType>;
    using typename base_t::value_t;
    using typename base_t::index_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::colmat_value_t;
    using typename base_t::rowmat_value_t;
    using typename base_t::sp_mat_value_t;
    using string_t = std::string;
    using io_t = io::IOSNPUnphased<MmapPtrType>;
    
protected:
    static constexpr value_t _max = std::numeric_limits<value_t>::max();
    const io_t& _io;             // IO handler
    const size_t _n_threads;    // number of threads
    vec_index_t _ibuff;
    vec_value_t _vbuff;
    vec_value_t _buff;

    ADELIE_CORE_STRONG_INLINE
    value_t _cmul(
        int j, 
        const Eigen::Ref<const vec_value_t>& v,
        const Eigen::Ref<const vec_value_t>& weights,
        size_t n_threads
    ) 
    {
        return snp_unphased_dot(
            [](auto x) { return x; },
            _io, j, v * weights, n_threads, _buff
        );
    }

    ADELIE_CORE_STRONG_INLINE
    void _ctmul(
        int j, 
        value_t v, 
        Eigen::Ref<vec_value_t> out,
        size_t n_threads
    ) const
    {
        snp_unphased_axi(_io, j, v, out, n_threads);
    }

public:
    explicit MatrixNaiveSNPUnphased(
        const io_t& io,
        size_t n_threads
    ): 
        _io(io),
        _n_threads(n_threads),
        _ibuff(_io.rows()),
        _vbuff(_io.rows()),
        _buff(n_threads)
    {
        if (n_threads < 1) {
            throw util::adelie_core_error("n_threads must be >= 1.");
        }
        _vbuff.setConstant(_max);
    }

    value_t cmul(
        int j, 
        const Eigen::Ref<const vec_value_t>& v,
        const Eigen::Ref<const vec_value_t>& weights
    ) override
    {
        base_t::check_cmul(j, v.size(), weights.size(), rows(), cols());
        return _cmul(j, v, weights, _n_threads);
    }

    void ctmul(
        int j, 
        value_t v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_ctmul(j, out.size(), rows(), cols());
        _ctmul(j, v, out, _n_threads);
    }

    void bmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_bmul(j, q, v.size(), weights.size(), out.size(), rows(), cols());
        for (int t = 0; t < q; ++t) {
            out[t] = _cmul(j + t, v, weights, _n_threads);
        }
    }

    void btmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_btmul(j, q, v.size(), out.size(), rows(), cols());
        for (int t = 0; t < q; ++t) {
            _ctmul(j + t, v[t], out, _n_threads);
        }
    }

    void mul(
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        const auto routine = [&](int t) {
            out[t] = _cmul(t, v, weights, 1);
        };
        if (_n_threads <= 1) {
            for (int t = 0; t < cols(); ++t) routine(t);
        } else {
            #pragma omp parallel for schedule(static) num_threads(_n_threads)
            for (int t = 0; t < cols(); ++t) routine(t);
        }
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

        for (int i1 = 0; i1 < q; ++i1) 
        {
            const auto index_1 = j+i1;
            const value_t imp_1 = _io.impute()[index_1];

            // if covariance must be computed,
            // cache index_1 information. 
            size_t nnz = 0;
            if (i1) {
                for (int c = 0; c < io_t::n_categories; ++c) {
                    auto it = _io.begin(index_1, c);
                    const auto end = _io.end(index_1, c);
                    const value_t val = (c == 0) ? imp_1 : c;
                    for (; it != end; ++it) {
                        const auto idx = *it;
                        _vbuff[idx] = val;
                        _ibuff[nnz] = idx;
                        ++nnz;
                    }
                }
            } 

            for (int i2 = 0; i2 <= i1; ++i2) {
                if (i1 == i2) {
                    out(i1, i1) = snp_unphased_dot(
                        [](auto x) { return x * x; },
                        _io, 
                        index_1, 
                        sqrt_weights.square(), 
                        _n_threads, 
                        _buff
                    );
                    continue;
                }
                const auto index_2 = j+i2;
                out(i1, i2) = snp_unphased_dot(
                    [](auto x) { return x; },
                    _io, 
                    index_2,
                    sqrt_weights.square() * (
                        (_vbuff != _max).template cast<value_t>() * _vbuff
                    ),
                    _n_threads,
                    _buff
                );
            }

            // keep invariance by populating with inf
            for (size_t i = 0; i < nnz; ++i) {
                _vbuff[_ibuff[i]] = _max;
            }
        }

        for (int i1 = 0; i1 < q; ++i1) {
            for (int i2 = i1+1; i2 < q; ++i2) {
                out(i1, i2) = out(i2, i1);
            }
        }
    }

    int rows() const override { return _io.rows(); }
    int cols() const override { return _io.cols(); }

    void sp_btmul(
        const sp_mat_value_t& v,
        Eigen::Ref<rowmat_value_t> out
    ) override
    {
        base_t::check_sp_btmul(
            v.rows(), v.cols(), out.rows(), out.cols(), rows(), cols()
        );
        const auto routine = [&](int k) {
            typename sp_mat_value_t::InnerIterator it(v, k);
            auto out_k = out.row(k);
            out_k.setZero();
            for (; it; ++it) {
                _ctmul(it.index(), it.value(), out_k, 1);
            }
        };
        if (_n_threads <= 1) {
            for (int k = 0; k < v.outerSize(); ++k) routine(k);
        } else {
            #pragma omp parallel for schedule(static) num_threads(_n_threads)
            for (int k = 0; k < v.outerSize(); ++k) routine(k);
        }
    }
};

} // namespace matrix
} // namespace adelie_core