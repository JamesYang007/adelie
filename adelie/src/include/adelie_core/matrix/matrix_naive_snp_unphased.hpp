#pragma once
#include <cstdio>
#include <string>
#include <vector>
#include <adelie_core/matrix/utils.hpp>
#include <adelie_core/io/io_snp_unphased.hpp>

namespace adelie_core {
namespace matrix {

template <class ValueType>
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
    using io_t = io::IOSNPUnphased;
    using dyn_vec_io_t = std::vector<io_t>;
    
protected:
    const string_t _filename;   // filename because why not? :)
    const io_t _io;             // IO handler
    const size_t _n_threads;    // number of threads

    static auto init_io(
        const string_t& filename
    )
    {
        io_t io(filename);
        io.read();
        return io;
    }

public:
    explicit MatrixNaiveSNPUnphased(
        const string_t& filename,
        size_t n_threads
    ): 
        _filename(filename),
        _io(init_io(filename)),
        _n_threads(n_threads)
    {}

    value_t cmul(
        int j, 
        const Eigen::Ref<const vec_value_t>& v,
        const Eigen::Ref<const vec_value_t>& weights
    ) override
    {
        base_t::check_cmul(j, v.size(), weights.size(), rows(), cols());
        const auto inner = _io.inner(j);
        const auto value = _io.value(j);
        return spddot(inner, value, v * weights);
    }

    void ctmul(
        int j, 
        value_t v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_ctmul(j, out.size(), rows(), cols());
        const auto inner = _io.inner(j);
        const auto value = _io.value(j);

        dvzero(out, _n_threads);

        for (int i = 0; i < inner.size(); ++i) {
            out[inner[i]] = v * value[i];
        }
    }

    void bmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_bmul(j, q, v.size(), weights.size(), out.size(), rows(), cols());
        #pragma omp parallel for schedule(static) num_threads(_n_threads)
        for (int t = 0; t < q; ++t) 
        {
            const auto inner = _io.inner(j+t);
            const auto value = _io.value(j+t);
            out[t] = spddot(inner, value, v * weights);
        }
    }

    void btmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_btmul(j, q, v.size(), out.size(), rows(), cols());
        dvzero(out, _n_threads);
        for (int t = 0; t < q; ++t) 
        {
            const auto inner = _io.inner(j+t);
            const auto value = _io.value(j+t);
            for (int i = 0; i < inner.size(); ++i) {
                out[inner[i]] += value[i] * v[t];
            } 
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

        #pragma omp parallel for schedule(static) num_threads(_n_threads)
        for (int i1 = 0; i1 < q; ++i1) {
            for (int i2 = 0; i2 <= i1; ++i2) {
                const auto index_1 = j+i1;
                const auto index_2 = j+i2;
                const auto inner_1 = _io.inner(index_1);
                const auto inner_2 = _io.inner(index_2);
                const auto value_1 = _io.value(index_1);
                const auto value_2 = _io.value(index_2);

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
        #pragma omp parallel for schedule(static) num_threads(_n_threads)
        for (int k = 0; k < v.outerSize(); ++k) {
            typename sp_mat_value_t::InnerIterator it(v, k);
            auto out_k = out.row(k);
            out_k.setZero();
            for (; it; ++it) 
            {
                const auto t = it.index();
                const auto inner = _io.inner(t);
                const auto value = _io.value(t);
                for (int i = 0; i < inner.size(); ++i) {
                    out_k[inner[i]] += value[i] * it.value();
                } 
            }
        }
    }
};

} // namespace matrix
} // namespace adelie_core