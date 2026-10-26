#pragma once
#include <adelie_core/io/io_snp_unphased.hpp>
#include <vector>

#ifndef ADELIE_CORE_IO_SNP_COMBINE_S_TP
#define ADELIE_CORE_IO_SNP_COMBINE_S_TP \
    template <class MmapPtrType>
#endif
#ifndef ADELIE_CORE_IO_SNP_COMBINE_S
#define ADELIE_CORE_IO_SNP_COMBINE_S \
    IOSNPCombineS<MmapPtrType>
#endif

namespace adelie_core {
namespace io {

template <class MmapPtrType=std::unique_ptr<char, std::function<void(char*)>>>
class IOSNPCombineS : public IOSNPBase<MmapPtrType>
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
    // Use 128 chunk size to guarantee chunk_nnz fits in uint8 (nnz-1)
    static constexpr size_t chunk_size = (
        static_cast<size_t>(1UL) << (n_bits_per_byte * sizeof(chunk_inner_t))
    ) / 2; // 256/2 = 128

protected:
    static constexpr size_t _max_inner = (
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
    outer_t _ancestries; // number of ancestry labels A
    outer_t _cols;       // s * (2*A)
    vec_outer_t _nnz;    // per column sample nnz
    vec_outer_t _outer;  // per snp block offset

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

    outer_t ancestries() const {
        if (!_is_read) throw_no_read();
        return _ancestries;
    }

    outer_t cols() const {
        if (!_is_read) throw_no_read();
        return _cols;
    }

    Eigen::Ref<const vec_outer_t> nnz() const {
        if (!_is_read) throw_no_read();
        return _nnz;
    }

    Eigen::Ref<const vec_outer_t> outer() const {
        if (!_is_read) throw_no_read();
        return _outer;
    }

    Eigen::Ref<const buffer_t> col(int j) const {
        return Eigen::Map<const buffer_t>(
            _buffer.data() + _outer[j],
            _outer[j+1] - _outer[j]
        );
    }

    // Column within block: 0..A-1 => mutated-haplotypes-per-ancestry
    //                      A..2A-1 => ancestry-dosage-per-ancestry
    const char* col_k(int j, int k) const {
        const auto _col = col(j);
        return (
            _col.data() +
            internal::read_as<outer_t>(_col.data() + sizeof(outer_t) * k)
        );
    }

    inner_t n_chunks(int j, int k) const {
        const auto* _col_k = col_k(j, k);
        return internal::read_as<inner_t>(_col_k);
    }

    iterator begin(int j, int k, int chunk) const {
        return iterator(chunk, col_k(j, k));
    }

    iterator begin(int j, int k) const { return begin(j, k, 0); }
    iterator end(int j, int k) const { return begin(j, k, n_chunks(j, k)); }

    rowarr_value_t to_dense(
        size_t n_threads
    ) const;

    std::tuple<size_t, std::unordered_map<std::string, double>> write(
        const Eigen::Ref<const colarr_value_t>& calldata,
        const Eigen::Ref<const colarr_value_t>& ancestries,
        size_t A,
        size_t n_threads
    ) const;

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


