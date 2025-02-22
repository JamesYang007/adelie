#include "py_decl.hpp"
#include <adelie_core/io/io_snp_phased_ancestry.hpp>
#include <adelie_core/io/io_snp_unphased.hpp>
#include <adelie_core/matrix/utils.hpp>
#include <adelie_core/util/stopwatch.hpp>

namespace py = pybind11;
namespace ad = adelie_core;

template <class ValueType = double, class IndexType=Eigen::Index>
void utils(py::module_& m)
{
    using value_t = ValueType;
    using index_t = IndexType;
    using vec_index_t = ad::util::rowvec_type<index_t>;
    using vec_value_t = ad::util::rowvec_type<value_t>;
    using mvec_value_t = Eigen::Matrix<value_t, 1, Eigen::Dynamic, Eigen::RowMajor>;
    using rowmat_value_t = ad::util::rowmat_type<value_t>;
    using colmat_value_t = ad::util::colmat_type<value_t>;
    using ref_rowarr_value_t = Eigen::Ref<ad::util::rowarr_type<value_t>>;
    using ref_rowmat_value_t = Eigen::Ref<rowmat_value_t>;
    using ref_vec_value_t = Eigen::Ref<vec_value_t>;
    using ref_mvec_value_t = Eigen::Ref<mvec_value_t>;
    using cref_vec_index_t = Eigen::Ref<const vec_index_t>;
    using cref_vec_value_t = Eigen::Ref<const vec_value_t>;
    using cref_rowarr_value_t = Eigen::Ref<const ad::util::rowarr_type<value_t>>;
    using cref_colmat_value_t = Eigen::Ref<const colmat_value_t>;
    using cref_mvec_value_t = Eigen::Ref<const Eigen::Matrix<value_t, 1, Eigen::Dynamic, Eigen::RowMajor>>;
    using snp_unphased_io_t = ad::io::IOSNPUnphased<>;
    using snp_phased_ancestry_io_t = ad::io::IOSNPPhasedAncestry<>;
    using sw_t = ad::util::Stopwatch;

    m.def("dvaddi", ad::matrix::dvaddi<ref_vec_value_t, cref_vec_value_t>);
    m.def("dmmeq", ad::matrix::dmmeq<ref_rowarr_value_t, cref_rowarr_value_t>);
    m.def("dvzero", ad::matrix::dvzero<ref_vec_value_t>);
    m.def("ddot", ad::matrix::ddot<cref_mvec_value_t, cref_mvec_value_t, ref_vec_value_t>);
    m.def("dgemv", ad::matrix::dgemv<ad::util::operator_type::_eq, cref_colmat_value_t, cref_mvec_value_t, ref_rowmat_value_t, ref_mvec_value_t>);

    m.def("bench_dvaddi", [](
        ref_vec_value_t x1,
        cref_vec_value_t x2,
        size_t n_threads,
        size_t n_sims
    ) {
        sw_t sw;
        double time_elapsed = 0;
        for (size_t i = 0; i < n_sims; ++i) {
            sw.start();
            ad::matrix::dvaddi(x1, x2, n_threads);
            time_elapsed += sw.elapsed();
        }
        return time_elapsed / n_sims;
    });

    m.def("bench_dvsubi", [](
        ref_vec_value_t x1,
        cref_vec_value_t x2,
        size_t n_threads,
        size_t n_sims
    ) {
        sw_t sw;
        double time_elapsed = 0;
        for (size_t i = 0; i < n_sims; ++i) {
            sw.start();
            ad::matrix::dvsubi(x1, x2, n_threads);
            time_elapsed += sw.elapsed();
        }
        return time_elapsed / n_sims;
    });

    m.def("bench_dvveq", [](
        ref_vec_value_t x1,
        cref_vec_value_t x2,
        size_t n_threads,
        size_t n_sims
    ) {
        sw_t sw;
        double time_elapsed = 0;
        for (size_t i = 0; i < n_sims; ++i) {
            sw.start();
            ad::matrix::dvveq(x1, x2, n_threads);
            time_elapsed += sw.elapsed();
        }
        return time_elapsed / n_sims;
    });

    m.def("bench_dvzero", [](
        ref_vec_value_t x1,
        cref_vec_value_t,
        size_t n_threads,
        size_t n_sims
    ) {
        sw_t sw;
        double time_elapsed = 0;
        for (size_t i = 0; i < n_sims; ++i) {
            sw.start();
            ad::matrix::dvzero(x1, n_threads);
            time_elapsed += sw.elapsed();
        }
        return time_elapsed / n_sims;
    });

    m.def("bench_ddot", [](
        cref_mvec_value_t x1,
        cref_mvec_value_t x2,
        size_t n_threads,
        size_t n_sims
    ) {
        sw_t sw;
        double time_elapsed = 0;
        volatile double output = 0;
        vec_value_t buff(n_threads);
        for (size_t i = 0; i < n_sims; ++i) {
            sw.start();
            output += ad::matrix::ddot(x1, x2, n_threads, buff);
            time_elapsed += sw.elapsed();
        }
        return time_elapsed / n_sims;
    });

    m.def("bench_ddot_opt", [](
        cref_mvec_value_t x1,
        cref_mvec_value_t x2,
        size_t n_threads,
        size_t n_sims
    ) {
        sw_t sw;
        double time_elapsed = 0;
        vec_value_t buff(n_threads);
        sw.start();
        #pragma omp parallel num_threads(n_threads)
        {
            for (size_t i = 0; i < n_sims; ++i) {
                const size_t n = x1.size();
                const int n_blocks = std::min(n_threads, n);
                const int block_size = n / n_blocks;
                const int remainder = n % n_blocks;

                // Test option: try adding nowait and see performance difference.
                #pragma omp for schedule(static) 
                for (int t = 0; t < n_blocks; ++t)
                {
                    const auto begin = (
                        std::min<int>(t, remainder) * (block_size + 1) 
                        + std::max<int>(t-remainder, 0) * block_size
                    );
                    const auto size = block_size + (t < remainder);
                    buff[t] = x1.segment(begin, size).dot(x2.segment(begin, size));
                }
            }
        }
        time_elapsed += sw.elapsed();
        return time_elapsed / n_sims;
    });

    m.def("bench_dmmeq", [](
        ref_rowarr_value_t x1,
        cref_rowarr_value_t x2,
        size_t n_threads,
        size_t n_sims
    ) {
        sw_t sw;
        double time_elapsed = 0;
        for (size_t i = 0; i < n_sims; ++i) {
            sw.start();
            ad::matrix::dmmeq(x1, x2, n_threads);
            time_elapsed += sw.elapsed();
        }
        return time_elapsed / n_sims;
    });
    
    m.def("bench_dgemv_eq", [](
        cref_colmat_value_t x1, 
        cref_colmat_value_t x2, 
        size_t n_threads,
        size_t n_sims
    ) {
        rowmat_value_t buff(n_threads, x1.cols());
        mvec_value_t out(x1.cols());
        out.setZero();
        auto v = x2.col(0).transpose();
        sw_t sw;
        double time_elapsed = 0;
        for (size_t i = 0; i < n_sims; ++i) {
            sw.start();
            ad::matrix::dgemv<ad::util::operator_type::_eq>(x1, v, n_threads, buff, out);
            time_elapsed += sw.elapsed();
        }
        return time_elapsed / n_sims;
    });

    m.def("bench_dgemv_add", [](
        cref_colmat_value_t x1, 
        cref_colmat_value_t x2, 
        size_t n_threads,
        size_t n_sims
    ) {
        rowmat_value_t buff(n_threads, x1.cols());
        mvec_value_t out(x1.cols());
        out.setZero();
        auto v = x2.col(0).transpose();
        sw_t sw;
        double time_elapsed = 0;
        for (size_t i = 0; i < n_sims; ++i) {
            sw.start();
            ad::matrix::dgemv<ad::util::operator_type::_add>(x1, v, n_threads, buff, out);
            time_elapsed += sw.elapsed();
        }
        return time_elapsed / n_sims;
    });

    m.def("bench_sq_norm", [](
        cref_colmat_value_t m, 
        size_t n_threads,
        size_t n_sims
    ) {
        mvec_value_t out(m.cols());
        sw_t sw;
        double time_elapsed = 0;
        for (size_t i = 0; i < n_sims; ++i) {
            sw.start();
            ad::matrix::sq_norm(m, out, n_threads);
            time_elapsed += sw.elapsed();
        }
        return time_elapsed / n_sims;
    });

    m.def("bench_spddot", [](
        cref_vec_index_t inner,
        cref_vec_value_t value,
        cref_vec_value_t x,
        size_t n_threads,
        size_t n_sims
    ) {
        sw_t sw;
        double time_elapsed = 0;
        volatile double output = 0;
        vec_value_t buff(n_threads);
        for (size_t i = 0; i < n_sims; ++i) {
            sw.start();
            output += ad::matrix::spddot(inner, value, x, n_threads, buff);
            time_elapsed += sw.elapsed();
        }
        return time_elapsed / n_sims;
    });

    m.def("bench_spaxi", [](
        cref_vec_index_t inner,
        cref_vec_value_t value,
        cref_vec_value_t x,
        size_t n_threads,
        size_t n_sims
    ) {
        sw_t sw;
        double time_elapsed = 0;
        vec_value_t out(x.size());
        out.setZero();
        for (size_t i = 0; i < n_sims; ++i) {
            sw.start();
            ad::matrix::spaxi(inner, value, x[0], out, n_threads);
            time_elapsed += sw.elapsed();
        }
        return time_elapsed / n_sims;
    });

    m.def("bench_snp_unphased_dot", [](
        const snp_unphased_io_t& io,
        int j, 
        cref_vec_value_t& v,
        size_t n_threads,
        size_t n_sims
    ){
        sw_t sw;
        double time_elapsed = 0;
        volatile double out = 0;
        vec_value_t buff(n_threads);
        for (size_t i = 0; i < n_sims; ++i) {
            sw.start();
            out += ad::matrix::snp_unphased_dot(
                [](auto x) { return x; }, io, j, v, n_threads, buff
            );
            time_elapsed += sw.elapsed();
        }
        return time_elapsed / n_sims;
    });

    m.def("bench_snp_unphased_axi", [](
        const snp_unphased_io_t& io,
        int j, 
        cref_vec_value_t& v,
        size_t n_threads,
        size_t n_sims
    ){
        sw_t sw;
        double time_elapsed = 0;
        vec_value_t out(v.size()); 
        out.setZero();
        for (size_t i = 0; i < n_sims; ++i) {
            sw.start();
            ad::matrix::snp_unphased_axi(
                io, j, v[0], out, n_threads
            );
            time_elapsed += sw.elapsed();
        }
        return time_elapsed / n_sims;
    });

    m.def("bench_snp_phased_ancestry_dot", [](
        const snp_phased_ancestry_io_t& io,
        int j, 
        cref_vec_value_t& v,
        size_t n_threads,
        size_t n_sims
    ){
        sw_t sw;
        double time_elapsed = 0;
        volatile double out = 0;
        vec_value_t buff(n_threads);
        for (size_t i = 0; i < n_sims; ++i) {
            sw.start();
            out += ad::matrix::snp_phased_ancestry_dot(
                io, j, v, n_threads, buff
            );
            time_elapsed += sw.elapsed();
        }
        return time_elapsed / n_sims;
    });

    m.def("bench_snp_phased_ancestry_axi", [](
        const snp_phased_ancestry_io_t& io,
        int j, 
        cref_vec_value_t& v,
        size_t n_threads,
        size_t n_sims
    ){
        sw_t sw;
        double time_elapsed = 0;
        vec_value_t out(v.size());
        for (size_t i = 0; i < n_sims; ++i) {
            sw.start();
            ad::matrix::snp_phased_ancestry_axi(
                io, j, v[0], out, n_threads
            );
            time_elapsed += sw.elapsed();
        }
        return time_elapsed / n_sims;
    });
}

void register_matrix_utils(py::module_& m)
{
    utils(m);
}