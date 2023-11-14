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
        const auto A = ancestries();
        out.setZero();

        int begin = 0;
        while (begin < q)
        {
            const auto snp = (j + begin) / A;
            const auto slice = _io_slice_map[snp];
            const auto& io = _ios[slice];
            const auto index = _io_index_map[snp];
            const auto ancestry_lower = (j + begin) % A;
            const auto ancestry_upper = std::min(ancestry_lower + q, A);

            if (ancestry_lower == 0 && ancestry_upper == A) {
                for (int hap = 0; hap < 2; ++hap) {
                    const auto inner = io.inner(index, hap);
                    const auto ancestry = io.ancestry(index, hap);
                    for (int i = 0; i < inner.size(); ++i) {
                        out[begin+ancestry[i]] += v[inner[i]];
                    }
                }
            } else {
                for (int hap = 0; hap < 2; ++hap) {
                    const auto inner = io.inner(index, hap);
                    const auto ancestry = io.ancestry(index, hap);
                    for (int i = 0; i < inner.size(); ++i) {
                        if (ancestry[i] < ancestry_lower || ancestry[i] >= ancestry_upper) continue;
                        out[begin+ancestry[i]-ancestry_lower] += v[inner[i]];
                    }
                }
            }

            begin += ancestry_upper - ancestry_lower;
        }     
    }

    void btmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        // TODO
    }

    void mul(
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {
        // TODO
    }

    void cov(
        int j, int q,
        const Eigen::Ref<const vec_value_t>& sqrt_weights,
        Eigen::Ref<colmat_value_t> out,
        Eigen::Ref<colmat_value_t> buffer
    ) const override
    {
        // TODO
    }

    int rows() const override { return _ios[0].rows(); }
    int cols() const override { return _snps * ancestries(); }

    void sp_btmul(
        int j, int q,
        const sp_mat_value_t& v,
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<rowmat_value_t> out
    ) const override
    {
        // TODO
    }

    void to_dense(
        int j, int q,
        Eigen::Ref<colmat_value_t> out
    ) const override
    {
        // TODO
    }

    void means(
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) const override
    {
        // TODO
    }
};

} // namespace matrix
} // namespace adelie_core