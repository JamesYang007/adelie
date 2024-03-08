#pragma once
#include <string>
#include <vector>
#include <adelie_core/matrix/matrix_naive_base.hpp>
#include <adelie_core/matrix/utils.hpp>
#include <adelie_core/io/io_snp_phased_ancestry.hpp>

namespace adelie_core {
namespace matrix {

template <class ValueType>
class MatrixNaiveSNPPhasedAncestry: public MatrixNaiveBase<ValueType>
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
    using io_t = io::IOSNPPhasedAncestry;
    using dyn_vec_io_t = std::vector<io_t>;
    
protected:
    const string_t _filename;   // filename because why not? :)
    const io_t _io;             // IO handler
    const size_t _n_threads;    // number of threads
    vec_value_t _vbuff;         // vector buffer
    util::rowarr_type<value_t> _buff;           // buffer

    static auto init_io(
        const string_t& filename
    )
    {
        io_t io(filename);
        io.read();
        return io;
    }

    static void throw_bad_start_index(int j, int A)
    {
        throw std::runtime_error(
            "Bad starting index " + std::to_string(j) +
            " with ancestries " + std::to_string(A)
        );
    }

    static void throw_bad_size(int q, int A)
    {
        throw std::runtime_error(
            "Bad size " + std::to_string(q) +
            " with ancestries " + std::to_string(A)
        );
    }

public:
    explicit MatrixNaiveSNPPhasedAncestry(
        const string_t& filename,
        size_t n_threads
    ): 
        _filename(filename),
        _io(init_io(filename)),
        _n_threads(n_threads),
        _vbuff(n_threads),
        _buff(n_threads, _io.ancestries())
    {}

    auto ancestries() const { return _io.ancestries(); }

    value_t cmul(
        int j, 
        const Eigen::Ref<const vec_value_t>& v,
        const Eigen::Ref<const vec_value_t>& weights
    ) override
    {
        base_t::check_cmul(j, v.size(), weights.size(), rows(), cols());

        const auto A = ancestries();
        const auto snp = j / A;
        const auto anc = j % A;

        value_t sum = 0;
        for (int hap = 0; hap < 2; ++hap) {
            const auto inner = _io.inner(snp, hap);
            const auto ancestry = _io.ancestry(snp, hap);

            for (int i = 0; i < inner.size(); ++i) {
                if (ancestry[i] != anc) continue;
                sum += v[inner[i]] * weights[inner[i]];
            }
        }

        return sum;
    }

    void ctmul(
        int j, 
        value_t v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_ctmul(j, out.size(), rows(), cols());

        const auto A = ancestries();
        const auto snp = j / A;
        const auto anc = j % A;

        dvzero(out, _n_threads);

        for (int hap = 0; hap < 2; ++hap) {
            const auto inner = _io.inner(snp, hap);
            const auto ancestry = _io.ancestry(snp, hap);
            for (int i = 0; i < inner.size(); ++i) {
                if (ancestry[i] != anc) continue;
                out[inner[i]] += v;
            }
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

        const int A = ancestries();

        out.setZero(); // don't parallelize! q is usually small

        int n_batches = (
            (j + q - A * (j / A) + A - 1) / A
        );

        #pragma omp parallel for schedule(static) num_threads(_n_threads)
        for (int b = 0; b < n_batches; ++b)
        {
            const auto n_solved =  (b > 0) * (A * ((j / A) + 1) - j + (b-1) * A);
            const auto begin = j + n_solved;
            const auto snp = begin / A;
            const auto ancestry_lower = begin % A;
            const auto ancestry_upper = std::min<int>(ancestry_lower + q - n_solved, A);

            const auto _common_routine = [&](const auto func) {
                for (int hap = 0; hap < 2; ++hap) {
                    const auto inner = _io.inner(snp, hap);
                    const auto ancestry = _io.ancestry(snp, hap);
                    func(inner, ancestry);
                }
            };

            // optimized routine when additional check is not required
            if (ancestry_lower == 0 && ancestry_upper == A) {
                _common_routine([&](const auto& inner, const auto& ancestry) {
                    for (int i = 0; i < inner.size(); ++i) {
                        out[n_solved+ancestry[i]] += v[inner[i]] * weights[inner[i]];
                    }
                });
            } else {
                _common_routine([&](const auto& inner, const auto& ancestry) {
                    for (int i = 0; i < inner.size(); ++i) {
                        if (ancestry[i] < ancestry_lower || ancestry[i] >= ancestry_upper) continue;
                        out[n_solved+ancestry[i]-ancestry_lower] += v[inner[i]] * weights[inner[i]];
                    }
                });
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

        const int A = ancestries();

        dvzero(out, _n_threads);

        int n_solved = 0;
        while (n_solved < q) 
        {
            const auto begin = j + n_solved;
            const auto snp = begin / A;
            const auto ancestry_lower = begin % A;
            const auto ancestry_upper = std::min<int>(ancestry_lower + q - n_solved, A);

            const auto _common_routine = [&](const auto func) {
                for (int hap = 0; hap < 2; ++hap) {
                    const auto inner = _io.inner(snp, hap);
                    const auto ancestry = _io.ancestry(snp, hap);
                    func(inner, ancestry);
                }
            };

            if (ancestry_lower == 0 && ancestry_upper == A) {
                _common_routine([&](const auto& inner, const auto& ancestry) {
                    for (int i = 0; i < inner.size(); ++i) {
                        out[inner[i]] += v[n_solved + ancestry[i]];
                    }
                });
            } else {
                _common_routine([&](const auto& inner, const auto& ancestry) {
                    for (int i = 0; i < inner.size(); ++i) {
                        if (ancestry[i] < ancestry_lower || ancestry[i] >= ancestry_upper) continue;
                        out[inner[i]] += v[n_solved + ancestry[i] - ancestry_lower];
                    }
                });
            }

            n_solved += ancestry_upper - ancestry_lower;
        }
    }

    void mul(
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        bmul(0, out.size(), v, weights, out);
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

        const auto A = ancestries();

        out.setZero(); // don't parallelize! q is usually small

        int n_solved0 = 0;
        while (n_solved0 < q) {
            const auto begin0 = j + n_solved0;
            const auto snp0 = begin0 / A;
            const auto ancestry_lower0 = begin0 % A;
            const auto ancestry_upper0 = std::min<int>(ancestry_lower0 + q - n_solved0, A);

            int n_solved1 = 0;
            while (n_solved1 <= n_solved0) {
                const auto begin1 = j + n_solved1;
                const auto snp1 = begin1 / A;
                const auto ancestry_lower1 = begin1 % A;
                const auto ancestry_upper1 = std::min<int>(ancestry_lower1 + q - n_solved1, A);

                for (int hap0 = 0; hap0 < 2; ++hap0) {
                    for (int hap1 = 0; hap1 < 2; ++hap1) {
                        const auto inner0 = _io.inner(snp0, hap0);
                        const auto ancestry0 = _io.ancestry(snp0, hap0);
                        const auto inner1 = _io.inner(snp1, hap1);
                        const auto ancestry1 = _io.ancestry(snp1, hap1);

                        int i0 = 0;
                        int i1 = 0;
                        while (
                            (i0 < inner0.size()) &&
                            (i1 < inner1.size())
                        ) {
                            while ((i0 < inner0.size()) && (inner0[i0] < inner1[i1])) ++i0;
                            if (i0 == inner0.size()) break;
                            while ((i1 < inner1.size()) && (inner1[i1] < inner0[i0])) ++i1;
                            if (i1 == inner1.size()) break;
                            while (
                                (i0 < inner0.size()) &&
                                (i1 < inner1.size()) &&
                                (inner0[i0] == inner1[i1])
                            ) {
                                if (
                                    ((n_solved0 == n_solved1) && (ancestry0[i0] < ancestry1[i1])) ||
                                    (ancestry0[i0] < ancestry_lower0) ||
                                    (ancestry0[i0] >= ancestry_upper0) ||
                                    (ancestry1[i1] < ancestry_lower1) ||
                                    (ancestry1[i1] >= ancestry_upper1)
                                ) {
                                    ++i0;
                                    ++i1;
                                    continue;
                                }
                                const auto k0 = n_solved0 + ancestry0[i0] - ancestry_lower0;
                                const auto k1 = n_solved1 + ancestry1[i1] - ancestry_lower1;
                                const auto sw = sqrt_weights[inner1[i1]];
                                out(k0, k1) += sw * sw;
                                ++i0;
                                ++i1;
                            }
                        }
                    }
                }

                n_solved1 += ancestry_upper1 - ancestry_lower1;
            }
            n_solved0 += ancestry_upper0 - ancestry_lower0;
        }     

        for (int i1 = 0; i1 < q; ++i1) {
            for (int i2 = i1+1; i2 < q; ++i2) {
                out(i1, i2) = out(i2, i1);
            }
        }
    }

    int rows() const override { return _io.rows(); }
    int cols() const override { return _io.snps() * ancestries(); }

    void sp_btmul(
        const sp_mat_value_t& v,
        Eigen::Ref<rowmat_value_t> out
    ) override
    {
        base_t::check_sp_btmul(
            v.rows(), v.cols(), out.rows(), out.cols(), rows(), cols()
        );
        const auto A = ancestries();
        #pragma omp parallel for schedule(static) num_threads(_n_threads)
        for (int k = 0; k < v.outerSize(); ++k) {
            typename sp_mat_value_t::InnerIterator it(v, k);
            auto out_k = out.row(k);
            out_k.setZero();
            for (; it; ++it) 
            {
                const auto t = it.index();
                const auto snp = t / A;
                const auto anc = t % A;

                for (int hap = 0; hap < 2; ++hap) {
                    const auto inner = _io.inner(snp, hap);
                    const auto ancestry = _io.ancestry(snp, hap);
                    for (int i = 0; i < inner.size(); ++i) {
                        if (ancestry[i] != anc) continue;
                        out_k[inner[i]] += it.value();
                    }
                }
            }
        }
    }
};

} // namespace matrix
} // namespace adelie_core