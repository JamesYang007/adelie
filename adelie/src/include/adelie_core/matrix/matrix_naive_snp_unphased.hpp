#pragma once
#include <cstdio>
#include <string>
#include <vector>
#include <adelie_core/matrix/matrix_naive_base.hpp>
#include <adelie_core/matrix/utils.hpp>
#include <adelie_core/io/io_snp_unphased.hpp>

namespace adelie_core {
namespace matrix {

template <class ValueType>
class MatrixNaiveSNPUnphased : public MatrixNaiveBase<ValueType>
{
public:
    using base_t = MatrixNaiveBase<ValueType>;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::colmat_value_t;
    using typename base_t::rowmat_value_t;
    using typename base_t::sp_mat_value_t;
    using io_t = io::IOSNPUnphased;
    using string_t = std::string;
    using dyn_vec_string_t = std::vector<string_t>;
    using dyn_vec_io_t = std::vector<io_t>;
    
protected:
    const dyn_vec_string_t _filenames;  // (F,) array of file names
    const dyn_vec_io_t _ios;            // (F,) array of IO handlers
    const size_t _p;                    // total number of feature across all slices
    const vec_index_t _io_slice_map;    // (p,) array mapping to matrix slice
    const vec_index_t _io_index_map;    // (p,) array mapping to (relative) index of the slice
    const size_t _n_threads;            // number of threads

    static auto init_ios(
        const dyn_vec_string_t& filenames
    )
    {
        dyn_vec_io_t ios;
        ios.reserve(filenames.size());
        for (int i = 0; i < filenames.size(); ++i) {
            ios.emplace_back(filenames[i]);
            ios.back().read();
        }
        return ios;
    }

    static auto init_p(
        const dyn_vec_io_t& ios
    )
    {
        size_t p = 0;
        for (const auto& io : ios) {
            p += io.cols();
        }
        return p;
    }

    static auto init_io_slice_map(
        const dyn_vec_io_t& ios,
        size_t p
    )
    {
        vec_index_t io_slice_map(p);
        size_t begin = 0;
        for (int i = 0; i < ios.size(); ++i) {
            const auto& io = ios[i];
            const auto pi = io.cols();
            for (int j = 0; j < pi; ++j) {
                io_slice_map[begin + j] = i;
            }
            begin += pi;
        } 
        return io_slice_map;
    }

    static auto init_io_index_map(
        const dyn_vec_io_t& ios,
        size_t p
    )
    {
        vec_index_t io_index_map(p);
        size_t begin = 0;
        for (int i = 0; i < ios.size(); ++i) {
            const auto& io = ios[i];
            const auto pi = io.cols();
            for (int j = 0; j < pi; ++j) {
                io_index_map[begin + j] = j;
            }
            begin += pi;
        } 
        return io_index_map;
    }

public:
    MatrixNaiveSNPUnphased(
        const dyn_vec_string_t& filenames,
        size_t n_threads
    ): 
        _filenames(filenames),
        _ios(init_ios(filenames)),
        _p(init_p(_ios)),
        _io_slice_map(init_io_slice_map(_ios, _p)),
        _io_index_map(init_io_index_map(_ios, _p)),
        _n_threads(n_threads)
    {
        if (filenames.size() == 0) {
            throw std::runtime_error(
                "filenames must be non-empty!"
            );
        }

        // make sure every file has the same number of rows.
        const size_t rows = _ios[0].rows();
        for (const auto& io : _ios) {
            if (io.rows() != rows) {
                throw std::runtime_error(
                    "Every slice must have same number of rows."
                );
            }
        }
    }

    value_t cmul(
        int j, 
        const Eigen::Ref<const vec_value_t>& v
    ) const override
    {
        const auto slice = _io_slice_map[j];
        const auto& io = _ios[slice];
        const auto index = _io_index_map[j];
        const auto inner = io.inner(index);
        const auto value = io.value(index);

        value_t sum = 0;
        for (int i = 0; i < inner.size(); ++i) {
            sum += v[inner[i]] * value[i];
        }

        return sum;
    }

    void ctmul(
        int j, 
        value_t v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) const override
    {
        const auto slice = _io_slice_map[j];
        const auto& io = _ios[slice];
        const auto index = _io_index_map[j];
        const auto inner = io.inner(index);
        const auto value = io.value(index);

        dvzero(out, _n_threads);

        for (int i = 0; i < inner.size(); ++i) {
            out[inner[i]] = v * value[i] * weights[inner[i]];
        }
    }

    void bmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {
        #pragma omp parallel for schedule(static) num_threads(_n_threads)
        for (int t = 0; t < q; ++t) 
        {
            const auto slice = _io_slice_map[j+t];
            const auto& io = _ios[slice];
            const auto index = _io_index_map[j+t];
            const auto inner = io.inner(index);
            const auto value = io.value(index);

            value_t sum = 0;
            for (int i = 0; i < inner.size(); ++i) {
                sum += v[inner[i]] * value[i];
            }
            out[t] = sum;
        }
    }

    void btmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        dvzero(out, _n_threads);
        for (int t = 0; t < q; ++t) 
        {
            const auto slice = _io_slice_map[j+t];
            const auto& io = _ios[slice];
            const auto index = _io_index_map[j+t];
            const auto inner = io.inner(index);
            const auto value = io.value(index);
            for (int i = 0; i < inner.size(); ++i) {
                out[inner[i]] += value[i] * weights[inner[i]] * v[t];
            } 
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
        Eigen::Ref<colmat_value_t> 
    ) const override
    {
        #pragma omp parallel for schedule(static) num_threads(_n_threads)
        for (int i1 = 0; i1 < q; ++i1) {
            for (int i2 = 0; i2 <= i1; ++i2) {
                const auto slice_1 = _io_slice_map[j+i1];
                const auto slice_2 = _io_slice_map[j+i2];
                const auto& io_1 = _ios[slice_1];
                const auto& io_2 = _ios[slice_2];
                const auto index_1 = _io_index_map[j+i1];
                const auto index_2 = _io_index_map[j+i2];
                const auto inner_1 = io_1.inner(index_1);
                const auto inner_2 = io_2.inner(index_2);
                const auto value_1 = io_1.value(index_1);
                const auto value_2 = io_2.value(index_2);

                out(i1, i2) = svsvwdot(
                    inner_1, value_1,
                    inner_2, value_2,
                    sqrt_weights.square()
                );
            }
        }
        for (int i1 = 0; i1 < q; ++i1) {
            for (int i2 = i1+1; i2 < q; ++i2) {
                out(i1, i2) = out(i2, i1);
            }
        }
    }

    int rows() const override { return _ios[0].rows(); }
    int cols() const override { return _p; }

    void sp_btmul(
        int j, int q,
        const sp_mat_value_t& v,
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<rowmat_value_t> out
    ) const override
    {
        #pragma omp parallel for schedule(static) num_threads(_n_threads)
        for (int k = 0; k < v.outerSize(); ++k) {
            typename sp_mat_value_t::InnerIterator it(v, k);
            auto out_k = out.row(k);
            out_k.setZero();
            for (; it; ++it) 
            {
                const auto t = it.index();
                const auto slice = _io_slice_map[j+t];
                const auto& io = _ios[slice];
                const auto index = _io_index_map[j+t];
                const auto inner = io.inner(index);
                const auto value = io.value(index);
                for (int i = 0; i < inner.size(); ++i) {
                    out_k[inner[i]] += value[i] * weights[inner[i]] * it.value();
                } 
            }
        }
    }

    void to_dense(
        int j, int q,
        Eigen::Ref<colmat_value_t> out
    ) const override
    {
        #pragma omp parallel for schedule(auto) num_threads(_n_threads)
        for (int k = 0; k < q; ++k) {
            const auto slice = _io_slice_map[j+k];
            const auto& io = _ios[slice];
            const auto index = _io_index_map[j+k];
            const auto inner = io.inner(index);
            const auto value = io.value(index);
            auto out_k = out.col(k);

            out_k.setZero();
            for (int i = 0; i < inner.size(); ++i) {
                out_k[inner[i]] = value[i];
            }
        }
    }

    void means(
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) const override
    {
        const auto p = cols();
        #pragma omp parallel for schedule(static) num_threads(_n_threads)
        for (int j = 0; j < p; ++j) 
        {
            const auto slice = _io_slice_map[j];
            const auto& io = _ios[slice];
            const auto index = _io_index_map[j];
            const auto inner = io.inner(index);
            const auto value = io.value(index);
            value_t sum = 0;
            for (int i = 0; i < inner.size(); ++i) {
                sum += weights[inner[i]] * value[i];
            }
            out[j] = sum;
        }
    }
};

} // namespace matrix
} // namespace adelie_core