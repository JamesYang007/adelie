#pragma once
#include <cstdio>
#include <string>
#include <vector>
#include <adelie_core/matrix/utils.hpp>
#include <adelie_core/io/io_snp_unphased.hpp>

namespace adelie_core {
namespace matrix {

template <class ValueType,
          class MmapPtrType=std::unique_ptr<char, std::function<void(char*)>>>
class MatrixNaiveSNPUnphased: public MatrixNaiveBase<ValueType>
{
public:
    using base_t = MatrixNaiveBase<ValueType>;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::colmat_value_t;
    using typename base_t::rowmat_value_t;
    using typename base_t::sp_mat_value_t;
    using string_t = std::string;
    using io_t = io::IOSNPUnphased<MmapPtrType>;
    
protected:
    const io_t _io;             // IO handler
    const size_t _n_threads;    // number of threads

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
    )
    {
        util::rowvec_type<value_t, 3> fills;
        fills[0] = _io.impute()[j];
        fills[1] = 1;
        fills[2] = 2;
        value_t sum = 0;
        for (int c = 0; c < 3; ++c) {
            auto it = _io.begin(c, j);
            const auto end = _io.end(c, j);
            value_t curr_sum = 0;
            for (; it != end; ++it) {
                const auto idx = *it;
                curr_sum += v[idx] * weights[idx]; 
            }
            sum += curr_sum * fills[c];
        }
        return sum;
    }

    ADELIE_CORE_STRONG_INLINE
    void _ctmul(
        int j, 
        value_t v, 
        Eigen::Ref<vec_value_t> out
    )
    {
        util::rowvec_type<value_t, 3> fills;
        fills[0] = _io.impute()[j];
        fills[1] = 1;
        fills[2] = 2;
        for (int c = 0; c < 3; ++c) {
            auto it = _io.begin(c, j);
            const auto end = _io.end(c, j);
            const auto curr_val = v * fills[c];
            for (; it != end; ++it) {
                const auto idx = *it;
                out[idx] += curr_val; 
            }
        }
    }

    template <class IterType, class WeightsType>
    value_t _svsvwdot(
        IterType it1,
        IterType end1,
        value_t v1,
        IterType it2,
        IterType end2,
        value_t v2,
        const WeightsType& weights
    )
    {
        value_t sum = 0;
        while (
            (it1 != end1) &&
            (it2 != end2)
        ) {
            const auto idx2 = *it2;
            while ((it1 != end1) && (*it1 < idx2)) ++it1;
            if (it1 == end1) break;
            const auto idx1 = *it1;
            while ((it2 != end2) && (*it2 < idx1)) ++it2;
            if (it2 == end2) break;
            while (
                (it1 != end1) &&
                (it2 != end2) &&
                (*it1 == *it2)
            ) {
                sum += weights[*it1];
                ++it1;
                ++it2;
            }
        }
        return sum * v1 * v2;
    }

    template <class IterType, class WeightsType>
    value_t _svsvwdot(
        IterType it1,
        IterType end1,
        IterType it2,
        IterType end2,
        const WeightsType& weights
    )
    {
        value_t sum = 0;
        while (
            (it1 != end1) &&
            (it2 != end2)
        ) {
            const auto idx2 = it2.index();
            while ((it1 != end1) && (it1.index() < idx2)) ++it1;
            if (it1 == end1) break;
            const auto idx1 = it1.index();
            while ((it2 != end2) && (it2.index() < idx1)) ++it2;
            if (it2 == end2) break;
            while (
                (it1 != end1) &&
                (it2 != end2) &&
                (it1.index() == it2.index())
            ) {
                sum += weights[it1.index()] * it1.value() * it2.value();
                ++it1;
                ++it2;
            }
        }
        return sum;
    }

    template <class WeightsType>
    value_t _svsvwdot(
        int j1,
        int j2,
        const WeightsType& weights
    )
    {
        using outer_t = typename io_t::outer_t;
        using inner_t = typename io_t::inner_t;
        using chunk_inner_t = typename io_t::chunk_inner_t;
        const auto col1 = _io.col(j1);
        const auto col2 = _io.col(j2);
        const auto imp1 = _io.impute()[j1];
        const auto imp2 = _io.impute()[j2];

        const auto routine = [&](
            const char* bf1, 
            value_t v1,
            const char* bf2,
            value_t v2
        ) -> value_t
        {
            const auto n_chunks1 = *reinterpret_cast<const inner_t*>(bf1);
            const auto n_chunks2 = *reinterpret_cast<const inner_t*>(bf2);
            size_t i1 = sizeof(inner_t);
            size_t i2 = sizeof(inner_t);
            if (!n_chunks1 || !n_chunks2) return 0;

            inner_t chunk_it1 = 0;
            inner_t chunk_idx1 = *reinterpret_cast<const inner_t*>(bf1+i1);
            i1 += sizeof(inner_t);
            inner_t chunk_nnz1 = (
                static_cast<inner_t>(*reinterpret_cast<const chunk_inner_t*>(bf1+i1))
                + 1
            );
            i1 += sizeof(chunk_inner_t);
            inner_t dense_chunk_index1 = chunk_idx1 * io_t::chunk_size;
            inner_t dense_index1 = (
                dense_chunk_index1 +
                *reinterpret_cast<const chunk_inner_t*>(bf1+i1)
            );
            inner_t inner1 = 0;

            inner_t chunk_it2 = 0;
            inner_t chunk_idx2 = *reinterpret_cast<const inner_t*>(bf2+i2);
            i2 += sizeof(inner_t);
            inner_t chunk_nnz2 = (
                static_cast<inner_t>(*reinterpret_cast<const chunk_inner_t*>(bf2+i2))
                + 1
            );
            i2 += sizeof(chunk_inner_t);
            inner_t dense_chunk_index2 = chunk_idx2 * io_t::chunk_size;
            inner_t dense_index2 = (
                dense_chunk_index2 +
                *reinterpret_cast<const chunk_inner_t*>(bf2+i2)
            );
            inner_t inner2 = 0;

            value_t sum = 0;

            while (
                (chunk_it1 != n_chunks1) &&
                (chunk_it2 != n_chunks2)
            )
            {
                if (dense_index1 < dense_index2) {
                    ++inner1;
                } else if (dense_index1 > dense_index2) {
                    ++inner2;
                } else {
                    sum += weights[dense_index1];
                    ++inner1;
                    ++inner2;
                }

                if (inner1 == chunk_nnz1) {
                    ++chunk_it1;
                    if (chunk_it1 == n_chunks1) break;

                    i1 += chunk_nnz1;
                    inner1 = 0;
                    chunk_idx1 = *reinterpret_cast<const inner_t*>(bf1+i1);
                    i1 += sizeof(inner_t);
                    chunk_nnz1 = (
                        static_cast<inner_t>(*reinterpret_cast<const chunk_inner_t*>(bf1+i1))
                        + 1
                    );
                    i1 += sizeof(chunk_inner_t);
                    dense_chunk_index1 = chunk_idx1 * io_t::chunk_size;
                } 
                dense_index1 = (
                    dense_chunk_index1 +
                    *reinterpret_cast<const chunk_inner_t*>(bf1+i1+inner1)
                );

                if (inner2 == chunk_nnz2) {
                    ++chunk_it2;
                    if (chunk_it2 == n_chunks2) break;

                    i2 += chunk_nnz2;
                    inner2 = 0;
                    chunk_idx2 = *reinterpret_cast<const inner_t*>(bf2+i2);
                    i2 += sizeof(inner_t);
                    chunk_nnz2 = (
                        static_cast<inner_t>(*reinterpret_cast<const chunk_inner_t*>(bf2+i2))
                        + 1
                    );
                    i2 += sizeof(chunk_inner_t);
                    dense_chunk_index2 = chunk_idx2 * io_t::chunk_size;
                } 
                dense_index2 = (
                    dense_chunk_index2 +
                    *reinterpret_cast<const chunk_inner_t*>(bf2+i2+inner2)
                );
            }
            return sum * v1 * v2;
        };

        value_t sum = 0;
        for (int c1 = 0; c1 < 3; ++c1) {
            const char* bf1 = (
                col1.data() + 
                reinterpret_cast<const outer_t*>(col1.data())[c1]
            );
            const value_t v1 = (c1 == 0) ? imp1 : c1;
            for (int c2 = 0; c2 < 3; ++c2) {
                const char* bf2 = (
                    col2.data() + 
                    reinterpret_cast<const outer_t*>(col2.data())[c2]
                );
                const value_t v2 = (c2 == 0) ? imp2 : c2;
                sum += routine(bf1, v1, bf2, v2);
            }
        }
        return sum;
    }

public:
    explicit MatrixNaiveSNPUnphased(
        const string_t& filename,
        const string_t& read_mode,
        size_t n_threads
    ): 
        _io(init_io(filename, read_mode)),
        _n_threads(n_threads)
    {
        if (n_threads < 1) {
            throw util::adelie_core_error("n_threads must be >= 1.");
        }
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
            for (int i2 = 0; i2 <= i1; ++i2) {
                const auto index_1 = j+i1;
                const auto index_2 = j+i2;

                if (i1 == i2) {
                    util::rowvec_type<value_t, 3> fills;
                    fills[0] = _io.impute()[index_1];
                    fills[0] *= fills[0];
                    fills[1] = 1;
                    fills[2] = 4;
                    value_t sum = 0;
                    for (int c = 0; c < 3; ++c) {
                        auto it = _io.begin(c, index_1);
                        const auto end = _io.end(c, index_1);
                        value_t curr_sum = 0;
                        for (; it != end; ++it) {
                            const auto idx = *it;
                            const auto val = sqrt_weights[idx];
                            curr_sum += val * val;
                        }
                        sum += curr_sum * fills[c]; 
                    }
                    out(i1, i2) = sum;
                    continue;
                }

                // TODO: performance is 3x slower than normal sparse-sparse dot.
                // Not sure why, but maybe too many invariance quantities?
                // All implementations below have the same runtime.
                // For now, keep the naive implementation.
                // This only matters in the rare case that the user needs
                // to run group lasso with this matrix, 
                // but this matrix is intended for lasso case.

                util::rowvec_type<value_t, 3> fills_1;
                util::rowvec_type<value_t, 3> fills_2;
                fills_1[0] = _io.impute()[index_1];
                fills_1[1] = 1;
                fills_1[2] = 2;
                fills_2[0] = _io.impute()[index_2];
                fills_2[1] = 1;
                fills_2[2] = 2;
                value_t sum = 0;
                for (int c1 = 0; c1 < 3; ++c1) {
                    auto it1 = _io.begin(c1, index_1);
                    const auto end1 = _io.end(c1, index_1);
                    const auto v1 = fills_1[c1];
                    for (int c2 = 0; c2 < 3; ++c2) {
                        auto it2 = _io.begin(c2, index_2);
                        const auto end2 = _io.end(c2, index_2);
                        const auto v2 = fills_2[c2];
                        sum += _svsvwdot(it1, end1, v1, it2, end2, v2, sqrt_weights.square());
                    }
                }
                out(i1, i2) = sum;

                //out(i1, i2) = _svsvwdot(
                //    _io.linear_begin(index_1),
                //    _io.linear_end(index_1),
                //    _io.linear_begin(index_2),
                //    _io.linear_end(index_2),
                //    sqrt_weights.square()
                //);

                //out(i1, i2) = _svsvwdot(index_1, index_2, sqrt_weights.square());
            }
        };
        if (_n_threads <= 1) {
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