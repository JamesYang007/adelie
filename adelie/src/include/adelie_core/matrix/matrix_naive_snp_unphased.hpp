#pragma once
#include <cstdio>
#include <string>
#include <vector>
#include <adelie_core/matrix/matrix_naive_snp_base.hpp>
#include <adelie_core/matrix/utils.hpp>
#include <adelie_core/io/io_snp_unphased.hpp>

namespace adelie_core {
namespace matrix {

template <class ValueType>
class MatrixNaiveSNPUnphased : 
    public MatrixNaiveBase<ValueType>,
    public MatrixNaiveSNPBase
{
public:
    using base_t = MatrixNaiveBase<ValueType>;
    using snp_base_t = MatrixNaiveSNPBase;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::colmat_value_t;
    using typename base_t::rowmat_value_t;
    using typename base_t::sp_mat_value_t;
    using typename snp_base_t::string_t;
    using typename snp_base_t::dyn_vec_string_t;
    using io_t = io::IOSNPUnphased;
    using dyn_vec_io_t = std::vector<io_t>;
    
protected:
    using snp_base_t::init_ios;
    using snp_base_t::init_snps;
    using snp_base_t::init_io_slice_map;
    using snp_base_t::init_io_index_map;

    const dyn_vec_io_t _ios;            // (F,) array of IO handlers
    const size_t _snps;                 // total number of SNPs across all slices
    const vec_index_t _io_slice_map;    // (s,) array mapping to matrix slice
    const vec_index_t _io_index_map;    // (s,) array mapping to (relative) index of the slice

public:
    MatrixNaiveSNPUnphased(
        const dyn_vec_string_t& filenames,
        size_t n_threads
    ): 
        snp_base_t(filenames, n_threads),
        _ios(init_ios<dyn_vec_io_t>(filenames)),
        _snps(init_snps(_ios)),
        _io_slice_map(init_io_slice_map(_ios, _snps)),
        _io_index_map(init_io_index_map(_ios, _snps))
    {
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
        base_t::check_cmul(j, v.size(), rows(), cols());
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
        base_t::check_ctmul(j, weights.size(), out.size(), rows(), cols());
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
        base_t::check_bmul(j, q, v.size(), out.size(), rows(), cols());
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
        base_t::check_btmul(j, q, v.size(), weights.size(), out.size(), rows(), cols());
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
        Eigen::Ref<colmat_value_t> buffer
    ) const override
    {
        base_t::check_cov(
            j, q, sqrt_weights.size(), 
            out.rows(), out.cols(), buffer.rows(), buffer.cols(), 
            rows(), cols()
        );
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
    int cols() const override { return _snps; }

    void sp_btmul(
        const sp_mat_value_t& v,
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<rowmat_value_t> out
    ) const override
    {
        base_t::check_sp_btmul(
            v.rows(), v.cols(), weights.size(), out.rows(), out.cols(), rows(), cols()
        );
        #pragma omp parallel for schedule(static) num_threads(_n_threads)
        for (int k = 0; k < v.outerSize(); ++k) {
            typename sp_mat_value_t::InnerIterator it(v, k);
            auto out_k = out.row(k);
            out_k.setZero();
            for (; it; ++it) 
            {
                const auto t = it.index();
                const auto slice = _io_slice_map[t];
                const auto& io = _ios[slice];
                const auto index = _io_index_map[t];
                const auto inner = io.inner(index);
                const auto value = io.value(index);
                for (int i = 0; i < inner.size(); ++i) {
                    out_k[inner[i]] += value[i] * weights[inner[i]] * it.value();
                } 
            }
        }
    }

    void means(
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) const override
    {
        base_t::check_means(weights.size(), out.size(), rows(), cols());
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