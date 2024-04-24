#pragma once
#include <cstdio>
#include <string>
#include <vector>
#include <adelie_core/matrix/utils.hpp>
#include <adelie_core/io/io_snp_unphased.hpp>
#include <omp.h>

namespace adelie_core {
namespace matrix {

template <class ValueType,
          class MmapPtrType=std::unique_ptr<char, std::function<void(char*)>>>
class MatrixNaiveSNPUnphased: public MatrixNaiveBase<ValueType>
{
public:
    using base_t = MatrixNaiveBase<ValueType>;
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
    static constexpr value_t _inf = std::numeric_limits<value_t>::infinity();
    const io_t _io;             // IO handler
    const size_t _n_threads;    // number of threads
    util::rowarr_type<index_t> _ibuff;
    util::rowarr_type<value_t> _vbuff;

    static auto init_io(
        const string_t& filename,
        const string_t& read_mode
    )
    {
        io_t io(filename, read_mode);
        io.read();
        return io;
    }

    ADELIE_CORE_STRONG_INLINE
    value_t _cmul(
        int j, 
        const Eigen::Ref<const vec_value_t>& v,
        const Eigen::Ref<const vec_value_t>& weights
    ) const
    {
        const value_t imp = _io.impute()[j];
        value_t sum = 0;
        for (int c = 0; c < io_t::n_categories; ++c) {
            auto it = _io.begin(j, c);
            const auto end = _io.end(j, c);
            const value_t val = (c == 0) ? imp : c;
            value_t curr_sum = 0;
            for (; it != end; ++it) {
                const auto idx = *it;
                curr_sum += v[idx] * weights[idx]; 
            }
            sum += curr_sum * val;
        }
        return sum;
    }

    ADELIE_CORE_STRONG_INLINE
    void _ctmul(
        int j, 
        value_t v, 
        Eigen::Ref<vec_value_t> out
    ) const
    {
        const value_t imp = _io.impute()[j];
        for (int c = 0; c < io_t::n_categories; ++c) {
            auto it = _io.begin(j, c);
            const auto end = _io.end(j, c);
            const value_t curr_val = v * ((c == 0) ? imp : c);
            for (; it != end; ++it) {
                const auto idx = *it;
                out[idx] += curr_val; 
            }
        }
    }

public:
    explicit MatrixNaiveSNPUnphased(
        const string_t& filename,
        const string_t& read_mode,
        size_t n_threads
    ): 
        _io(init_io(filename, read_mode)),
        _n_threads(n_threads),
        _ibuff(n_threads, _io.rows()),
        _vbuff(n_threads, _io.rows())
    {
        if (n_threads < 1) {
            throw util::adelie_core_error("n_threads must be >= 1.");
        }
        _vbuff.setConstant(_inf);
    }

    value_t cmul(
        int j, 
        const Eigen::Ref<const vec_value_t>& v,
        const Eigen::Ref<const vec_value_t>& weights
    ) override
    {
        base_t::check_cmul(j, v.size(), weights.size(), rows(), cols());
        return _cmul(j, v, weights);
    }

    void ctmul(
        int j, 
        value_t v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_ctmul(j, out.size(), rows(), cols());
        _ctmul(j, v, out);
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
            out[t] = _cmul(j + t, v, weights);
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
            _ctmul(j + t, v[t], out);
        }
    }

    void mul(
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        const auto routine = [&](int t) {
            out[t] = _cmul(t, v, weights);
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

        const auto routine = [&](int i1) {
            const auto thr_id = omp_get_thread_num();
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
                        _vbuff(thr_id, idx) = val;
                        _ibuff(thr_id, nnz) = idx;
                        ++nnz;
                    }
                }
            } 

            for (int i2 = 0; i2 <= i1; ++i2) {
                const auto index_2 = j+i2;

                if (i1 == i2) {
                    value_t sum = 0;
                    for (int c = 0; c < io_t::n_categories; ++c) {
                        auto it = _io.begin(index_1, c);
                        const auto end = _io.end(index_1, c);
                        const value_t val = (c == 0) ? imp_1 : c;
                        value_t curr_sum = 0;
                        for (; it != end; ++it) {
                            const auto idx = *it;
                            const auto val = sqrt_weights[idx];
                            curr_sum += val * val;
                        }
                        sum += curr_sum * val *val; 
                    }
                    out(i1, i2) = sum;
                    continue;
                }

                value_t sum = 0;
                const value_t imp_2 = _io.impute()[index_2];
                for (int c = 0; c < io_t::n_categories; ++c) {
                    auto it = _io.begin(index_2, c);
                    const auto end = _io.end(index_2, c);
                    const value_t val = (c == 0) ? imp_2 : c;
                    value_t curr_sum = 0;
                    for (; it != end; ++it) {
                        const auto idx = *it;
                        const auto v1 = _vbuff(thr_id, idx);
                        if (std::isinf(v1)) continue;
                        const auto sqrt_w = sqrt_weights[idx];
                        curr_sum += sqrt_w * sqrt_w * v1;
                    }
                    sum += curr_sum * val;
                }
                out(i1, i2) = sum;
            }

            // keep invariance by populating with inf
            for (size_t i = 0; i < nnz; ++i) {
                _vbuff(thr_id, _ibuff(thr_id, i)) = _inf;
            }
        };
        if ((_n_threads <= 1) || (q == 1)) {
            for (int i1 = 0; i1 < q; ++i1) routine(i1);
        } else {
            #pragma omp parallel for schedule(static) num_threads(_n_threads)
            for (int i1 = 0; i1 < q; ++i1) routine(i1);
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
                _ctmul(it.index(), it.value(), out_k);
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