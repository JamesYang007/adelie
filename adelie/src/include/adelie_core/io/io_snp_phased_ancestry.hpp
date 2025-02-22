#pragma once
#include <adelie_core/io/io_snp_unphased.hpp>

#ifndef ADELIE_CORE_IO_SNP_PHASED_ANCESTRY_TP
#define ADELIE_CORE_IO_SNP_PHASED_ANCESTRY_TP \
    template <class MmapPtrType>
#endif
#ifndef ADELIE_CORE_IO_SNP_PHASED_ANCESTRY
#define ADELIE_CORE_IO_SNP_PHASED_ANCESTRY \
    IOSNPPhasedAncestry<MmapPtrType>
#endif

namespace adelie_core {
namespace io {

template <class MmapPtrType=std::unique_ptr<char, std::function<void(char*)>>>
class IOSNPPhasedAncestry : public IOSNPBase<MmapPtrType>
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

    static constexpr size_t n_bits_per_byte = 8;
    static constexpr size_t chunk_size = (
        // casting helps MSVC with warning C4293
        static_cast<size_t>(1UL) << (n_bits_per_byte * sizeof(chunk_inner_t))
    );
    static constexpr size_t n_haps = 2;

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
    vec_outer_t _nnz0;
    vec_outer_t _nnz1;
    vec_outer_t _outer;

public:
    using iterator = IOSNPChunkIterator<
        chunk_size, inner_t, chunk_inner_t
    >;

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

    Eigen::Ref<const vec_outer_t> nnz0() const 
    {
        if (!_is_read) throw_no_read();
        return _nnz0;
    }

    Eigen::Ref<const vec_outer_t> nnz1() const 
    {
        if (!_is_read) throw_no_read();
        return _nnz1;
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

    const char* col_anc_hap(int j, int anc, int hap) const
    {
        const auto _col = col(j);
        const auto* _col_anc = (
            _col.data() + 
            internal::read_as<outer_t>(_col.data() + sizeof(outer_t) * anc)
        );
        return (
            _col_anc + 
            internal::read_as<outer_t>(_col_anc + sizeof(outer_t) * hap)
        );
    }

    inner_t n_chunks(int j, int anc, int hap) const
    {
        const auto* _col_anc_hap = col_anc_hap(j, anc, hap);
        return internal::read_as<inner_t>(_col_anc_hap);
    }

    iterator begin(int j, int anc, int hap, int chunk) const
    {
        return iterator(chunk, col_anc_hap(j, anc, hap));
    }

    iterator begin(int j, int anc, int hap) const
    {
        return begin(j, anc, hap, 0);
    }

    iterator end(int j, int anc, int hap) const
    {
        return begin(j, anc, hap, n_chunks(j, anc, hap));
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
};

} // namespace io
} // namespace adelie_core