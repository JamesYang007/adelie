#pragma once
#include <adelie_core/matrix/matrix_naive_base.hpp>
#include <adelie_core/matrix/matrix_naive_snp_base.hpp>
#include <adelie_core/matrix/utils.hpp>
#include <adelie_core/io/io_snp_phased_ancestry.hpp>

namespace adelie_core {
namespace matrix {

template <class ValueType>
class MatrixNaiveSNPPhasedAncestry : 
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
    using io_t = io::IOSNPPhasedAncestry;
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
    MatrixNaiveSNPPhasedAncestry(
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

        const size_t A = _ios[0].ancestries();
        for (const auto& io : _ios) {
            if (io.ancestries() != A) {
                throw std::runtime_error(
                    "Every slice must have same number of ancestries."
                );
            }
        }
    }

    auto ancestries() const { return _ios[0].ancestries(); }

    value_t cmul(
        int j, 
        const Eigen::Ref<const vec_value_t>& v
    ) const override
    {
        const auto A = ancestries();
        const auto snp = j / A;
        const auto anc = j % A;
        const auto slice = _io_slice_map[snp];
        const auto& io = _ios[slice];
        const auto index = _io_index_map[snp];

        value_t sum = 0;
        for (int hap = 0; hap < 2; ++hap) {
            const auto inner = io.inner(index, hap);
            const auto ancestry = io.ancestry(index, hap);
            for (int i = 0; i < inner.size(); ++i) {
                if (ancestry[i] != anc) continue;
                sum += v[inner[i]];
            }
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
        const auto A = ancestries();
        const auto snp = j / A;
        const auto anc = j % A;
        const auto slice = _io_slice_map[snp];
        const auto& io = _ios[slice];
        const auto index = _io_index_map[snp];

        dvzero(out, _n_threads);
        for (int hap = 0; hap < 2; ++hap) {
            const auto inner = io.inner(index, hap);
            const auto ancestry = io.ancestry(index, hap);
            for (int i = 0; i < inner.size(); ++i) {
                if (ancestry[i] != anc) continue;
                out[inner[i]] += v * weights[inner[i]];
            }
        }
    }

    void bmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {
        const int A = ancestries();
        out.setZero();

        int n_solved = 0;
        while (n_solved < q)
        {
            const auto begin = j + n_solved;
            const auto snp = begin / A;
            const auto slice = _io_slice_map[snp];
            const auto& io = _ios[slice];
            const auto index = _io_index_map[snp];
            const auto ancestry_lower = begin % A;
            const auto ancestry_upper = std::min<int>(ancestry_lower + q - n_solved, A);

            if (ancestry_lower == 0 && ancestry_upper == A) {
                for (int hap = 0; hap < 2; ++hap) {
                    const auto inner = io.inner(index, hap);
                    const auto ancestry = io.ancestry(index, hap);
                    for (int i = 0; i < inner.size(); ++i) {
                        out[n_solved+ancestry[i]] += v[inner[i]];
                    }
                }
            } else {
                for (int hap = 0; hap < 2; ++hap) {
                    const auto inner = io.inner(index, hap);
                    const auto ancestry = io.ancestry(index, hap);
                    for (int i = 0; i < inner.size(); ++i) {
                        if (ancestry[i] < ancestry_lower || ancestry[i] >= ancestry_upper) continue;
                        out[n_solved+ancestry[i]-ancestry_lower] += v[inner[i]];
                    }
                }
            }

            n_solved += ancestry_upper - ancestry_lower;
        }     
    }

    void btmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        const int A = ancestries();
        dvzero(out, _n_threads);

        int n_solved = 0;
        while (n_solved < q) 
        {
            const auto begin = j + n_solved;
            const auto snp = begin / A;
            const auto slice = _io_slice_map[snp];
            const auto& io = _ios[slice];
            const auto index = _io_index_map[snp];
            const auto ancestry_lower = begin % A;
            const auto ancestry_upper = std::min<int>(ancestry_lower + q - n_solved, A);

            if (ancestry_lower == 0 && ancestry_upper == A) {
                for (int hap = 0; hap < 2; ++hap) {
                    const auto inner = io.inner(index, hap);
                    const auto ancestry = io.ancestry(index, hap);
                    for (int i = 0; i < inner.size(); ++i) {
                        out[inner[i]] += weights[inner[i]] * v[n_solved + ancestry[i]];
                    }
                }
            } else {
                for (int hap = 0; hap < 2; ++hap) {
                    const auto inner = io.inner(index, hap);
                    const auto ancestry = io.ancestry(index, hap);
                    for (int i = 0; i < inner.size(); ++i) {
                        if (ancestry[i] < ancestry_lower || ancestry[i] >= ancestry_upper) continue;
                        out[inner[i]] += weights[inner[i]] * v[n_solved + ancestry[i] - ancestry_lower];
                    }
                }
            }

            n_solved += ancestry_upper - ancestry_lower;
        }
    }

    void mul(
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {
        bmul(0, out.size(), v, out);
    }

    void cov(
        int j, int q,
        const Eigen::Ref<const vec_value_t>& sqrt_weights,
        Eigen::Ref<colmat_value_t> out,
        Eigen::Ref<colmat_value_t> 
    ) const override
    {
        const auto A = ancestries();

        out.setZero();

        int n_solved0 = 0;
        while (n_solved0 < q) {
            const auto begin0 = j + n_solved0;
            const auto snp0 = begin0 / A;
            const auto slice0 = _io_slice_map[snp0];
            const auto& io0 = _ios[slice0];
            const auto index0 = _io_index_map[snp0];
            const auto ancestry_lower0 = begin0 % A;
            const auto ancestry_upper0 = std::min<int>(ancestry_lower0 + q - n_solved0, A);

            int n_solved1 = 0;
            while (n_solved1 <= n_solved0) {
                const auto begin1 = j + n_solved1;
                const auto snp1 = begin1 / A;
                const auto slice1 = _io_slice_map[snp1];
                const auto& io1 = _ios[slice1];
                const auto index1 = _io_index_map[snp1];
                const auto ancestry_lower1 = begin1 % A;
                const auto ancestry_upper1 = std::min<int>(ancestry_lower1 + q - n_solved1, A);

                for (int hap0 = 0; hap0 < 2; ++hap0) {
                    for (int hap1 = 0; hap1 < 2; ++hap1) {
                        const auto inner0 = io0.inner(index0, hap0);
                        const auto ancestry0 = io0.ancestry(index0, hap0);
                        const auto inner1 = io1.inner(index1, hap1);
                        const auto ancestry1 = io1.ancestry(index1, hap1);

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

    int rows() const override { return _ios[0].rows(); }
    int cols() const override { return _snps * ancestries(); }

    void sp_btmul(
        const sp_mat_value_t& v,
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<rowmat_value_t> out
    ) const override
    {
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
                const auto slice = _io_slice_map[snp];
                const auto& io = _ios[slice];
                const auto index = _io_index_map[snp];
                const auto anc = t % A;

                for (int hap = 0; hap < 2; ++hap) {
                    const auto inner = io.inner(index, hap);
                    const auto ancestry = io.ancestry(index, hap);
                    for (int i = 0; i < inner.size(); ++i) {
                        if (ancestry[i] != anc) continue;
                        out_k[inner[i]] += weights[inner[i]] * it.value();
                    }
                }
            }
        }
    }

    void means(
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) const override
    {
        dvzero(out, _n_threads);

        const auto A = ancestries();
        #pragma omp parallel for schedule(static) num_threads(_n_threads)
        for (int snp = 0; snp < _snps; ++snp) {
            const auto slice = _io_slice_map[snp];
            const auto& io = _ios[slice];
            const auto index = _io_index_map[snp];

            for (int hap = 0; hap < 2; ++hap) {
                const auto inner = io.inner(index, hap);
                const auto ancestry = io.ancestry(index, hap);
                for (int i = 0; i < inner.size(); ++i) {
                    out[A * snp + ancestry[i]] += weights[inner[i]];
                }
            }
        }
    }
};

} // namespace matrix
} // namespace adelie_core