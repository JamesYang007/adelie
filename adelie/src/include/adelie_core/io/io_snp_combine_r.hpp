#pragma once
#include <adelie_core/io/io_snp_unphased.hpp>
#include <vector>

#ifndef ADELIE_CORE_IO_SNP_COMBINE_R_TP
#define ADELIE_CORE_IO_SNP_COMBINE_R_TP \
    template <class MmapPtrType>
#endif
#ifndef ADELIE_CORE_IO_SNP_COMBINE_R
#define ADELIE_CORE_IO_SNP_COMBINE_R \
    IOSNPCombineR<MmapPtrType>
#endif

namespace adelie_core {
namespace io {

template <class MmapPtrType=std::unique_ptr<char, std::function<void(char*)>>>
class IOSNPCombineR : public IOSNPBase<MmapPtrType>
{
public:
    using base_t = IOSNPBase<MmapPtrType>;
    using outer_t = uint64_t;
    using inner_t = uint32_t;
    using chunk_inner_t = uint8_t;
    using value_t = int8_t;
    using vec_outer_t = util::rowvec_type<outer_t>;
    using vec_inner_t = util::rowvec_type<inner_t>;
    using vec_value_t = util::rowvec_type<value_t>;
    using rowarr_value_t = util::rowarr_type<value_t>;
    using colarr_value_t = util::colarr_type<value_t>;
    using typename base_t::bool_t;
    using typename base_t::buffer_t;

    // static constexpr size_t n_bits_per_byte = 8;
    // static constexpr size_t chunk_size = (
    //     // casting helps MSVC with warning C4293
    //     static_cast<size_t>(1UL) << (n_bits_per_byte * sizeof(chunk_inner_t))
    // );
    // NOTE: Each chunk stores at most one *byte* per allele occurrence
    // (dosage value 1) within a 128-sample window. In the worst case, where
    // all samples in the window have dosage == 2, we emit 256 bytes which
    // still fits in an unsigned 8-bit counter (encoded as *nnz − 1*).
    // Reducing the chunk size from 256→128 therefore guarantees that the
    // counter cannot overflow and avoids downstream buffer overruns that
    // manifested as segmentation faults for large cohorts (n≈10 k).
    static constexpr size_t chunk_size = 128;
    static constexpr size_t n_bits_per_byte = 8; // retained for compatibility

protected:
    static constexpr size_t _max_inner = (
        // casting helps MSVC with warning C4293
        static_cast<size_t>(1UL) << (n_bits_per_byte * sizeof(inner_t))
    );

    using base_t::throw_no_read;
    using base_t::fopen_safe;
    using base_t::is_big_endian;
    using base_t::_buffer;
    using base_t::_filename;
    using base_t::_is_read;

    outer_t _rows;
    outer_t _snps;
    outer_t _ancestries;
    outer_t _cols;
    vec_outer_t _nnz;
    vec_outer_t _outer;

public:
    using iterator = IOSNPChunkIterator<
        chunk_size, inner_t, chunk_inner_t
    >;

    //
    // Allow random-access "calls" so utils.hpp can do io(i,snp,anc)
    //
    inline value_t operator()(size_t sample,
                               size_t snp,
                               size_t anc) const
    {
        // scan the non-zero entries for (snp,anc)
        for (auto it = begin(snp, anc), e = end(snp, anc); it != e; ++it) {
            if (static_cast<size_t>(*it) == sample) {
                return value_t(1);
            }
        }
        return value_t(0);
    }

    using base_t::base_t;

    size_t read() override;

    outer_t rows() const {
        if (!_is_read) throw_no_read();
        return _rows;
    }

    outer_t snps() const {
        if (!_is_read) throw_no_read();
        return _snps;
    }

    outer_t ancestries() const 
    {
        if (!_is_read) throw_no_read();
        return _ancestries;
    }

    outer_t cols() const
    {
        if (!_is_read) throw_no_read();
        return _cols;
    }

    Eigen::Ref<const vec_outer_t> nnz() const 
    {
        if (!_is_read) throw_no_read();
        return _nnz;
    }

    Eigen::Ref<const vec_outer_t> outer() const
    {
        if (!_is_read) throw_no_read();
        return _outer;
    }

    Eigen::Ref<const buffer_t> col(int j) const
    {
        return Eigen::Map<const buffer_t>(
            _buffer.data() + _outer[j],
            _outer[j+1] - _outer[j]
        );
    }

    const char* col_anc(int j, int anc) const
    {
        const auto _col = col(j);
        return (
            _col.data() + 
            internal::read_as<outer_t>(_col.data() + sizeof(outer_t) * anc)
        );
    }

    inner_t n_chunks(int j, int anc) const
    {
        const auto* _col_anc = col_anc(j, anc);
        return internal::read_as<inner_t>(_col_anc);
    }

    iterator begin(int j, int anc, int chunk) const
    {
        return iterator(chunk, col_anc(j, anc));
    }

    iterator begin(int j, int anc) const
    {
        return begin(j, anc, 0);
    }

    iterator end(int j, int anc) const
    {
        return begin(j, anc, n_chunks(j, anc));
    }

    rowarr_value_t to_dense(
        size_t n_threads
    ) const;

    std::tuple<size_t, std::unordered_map<std::string, double>> write(
        const Eigen::Ref<const colarr_value_t>& calldata,
        const Eigen::Ref<const colarr_value_t>& ancestries,
        size_t A,
        size_t n_threads
    ) const;

    // Overload: write only a selected subset of ancestries.
    // - A_total is the total number of ancestries present in the input labels
    //   (upper bound; used for validation).
    // - selected_ancestries lists the ancestry labels (from the 0..A_total-1
    //   space) to be serialized. The output will have (1 + selected.size())
    //   columns per SNP block.
    std::tuple<size_t, std::unordered_map<std::string, double>> write(
        const Eigen::Ref<const colarr_value_t>& calldata,
        const Eigen::Ref<const colarr_value_t>& ancestries,
        size_t A_total,
        const std::vector<uint32_t>& selected_ancestries,
        size_t n_threads
    ) const;
};

} // namespace io
} // namespace adelie_core 