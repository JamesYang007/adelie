#pragma once
#include <adelie_core/io/io_snp_base.hpp>
#include <adelie_core/util/macros.hpp>

namespace adelie_core {
namespace io {
namespace internal {

template <class T>
ADELIE_CORE_STRONG_INLINE
T read_as(const void* ptr)
{
    T out;
    std::memcpy(&out, ptr, sizeof(T));
    return out;
}

} // namespace internal

template <size_t chunk_size, class InnerType, class ChunkInnerType>
struct IOSNPChunkIterator;

template <size_t chunk_size, class InnerType, class ChunkInnerType>
inline constexpr bool 
operator==(
    const IOSNPChunkIterator<chunk_size, InnerType, ChunkInnerType>& it1, 
    const IOSNPChunkIterator<chunk_size, InnerType, ChunkInnerType>& it2
)
{
    return it1.chunk_it == it2.chunk_it;
}

template <size_t chunk_size, class InnerType, class ChunkInnerType>
inline constexpr bool 
operator!=(
    const IOSNPChunkIterator<chunk_size, InnerType, ChunkInnerType>& it1, 
    const IOSNPChunkIterator<chunk_size, InnerType, ChunkInnerType>& it2
)
{
    return it1.chunk_it != it2.chunk_it;
}

template <size_t chunk_size, class InnerType, class ChunkInnerType>
struct IOSNPChunkIterator
{
    using inner_t = InnerType;
    using chunk_inner_t = ChunkInnerType;

    static_assert(sizeof(chunk_inner_t) == 1, "chunk_inner_t must be 1 byte.");

    inner_t chunk_it;
    const char* const ctg_buffer;
    const inner_t n_chunks;
    size_t buffer_idx = 0;
    inner_t chunk_index;
    inner_t chunk_nnz;
    inner_t inner;
    size_t dense_chunk_index;
    size_t dense_index;

    explicit IOSNPChunkIterator(
        inner_t chunk_it,
        const char* ctg_buffer
    ):
        chunk_it(chunk_it),
        ctg_buffer(ctg_buffer),
        n_chunks(internal::read_as<inner_t>(ctg_buffer))
    {
        if (chunk_it >= n_chunks) return;

        // increment past n_chunks and chunk_it number of chunks
        buffer_idx = sizeof(inner_t);
        for (inner_t i = 0; i < chunk_it; ++i) {
            buffer_idx += sizeof(inner_t);
            const size_t nnz = (
                static_cast<inner_t>(*reinterpret_cast<const chunk_inner_t*>(ctg_buffer + buffer_idx))
                + 1
            );
            buffer_idx += (
                sizeof(chunk_inner_t) + nnz * sizeof(chunk_inner_t)
            );
        }
        if (n_chunks) update();
    }

    ADELIE_CORE_STRONG_INLINE
    IOSNPChunkIterator& operator++() { 
        buffer_idx += sizeof(chunk_inner_t);
        ++inner;
        if (inner >= chunk_nnz) {
            ++chunk_it;
            if (chunk_it < n_chunks) update();
        } else {
            dense_index = (
                dense_chunk_index +
                *reinterpret_cast<const chunk_inner_t*>(ctg_buffer + buffer_idx)
            );
        }
        return *this; 
    }
    ADELIE_CORE_STRONG_INLINE
    size_t& operator*() { return dense_index; }
    friend constexpr bool operator==<>(const IOSNPChunkIterator&, 
                                       const IOSNPChunkIterator&);
    friend constexpr bool operator!=<>(const IOSNPChunkIterator&, 
                                       const IOSNPChunkIterator&);

    ADELIE_CORE_STRONG_INLINE
    void update()
    {
        chunk_index = internal::read_as<inner_t>(ctg_buffer + buffer_idx);
        buffer_idx += sizeof(inner_t);
        chunk_nnz = (
            static_cast<inner_t>(*reinterpret_cast<const chunk_inner_t*>(ctg_buffer + buffer_idx))
            + 1
        );
        buffer_idx += sizeof(chunk_inner_t);
        inner = 0;
        dense_chunk_index = chunk_index * chunk_size;
        dense_index = (
            dense_chunk_index +
            *reinterpret_cast<const chunk_inner_t*>(ctg_buffer + buffer_idx)
        );
    }
};

#ifndef ADELIE_CORE_IO_SNP_UNPHASED_TP
#define ADELIE_CORE_IO_SNP_UNPHASED_TP \
    template <class MmapPtrType>
#endif
#ifndef ADELIE_CORE_IO_SNP_UNPHASED
#define ADELIE_CORE_IO_SNP_UNPHASED \
    IOSNPUnphased<MmapPtrType>
#endif

template <class MmapPtrType=std::unique_ptr<char, std::function<void(char*)>>>
class IOSNPUnphased : public IOSNPBase<MmapPtrType>
{
public:
    using base_t = IOSNPBase<MmapPtrType>;
    using outer_t = uint64_t;
    using inner_t = uint32_t;
    using chunk_inner_t = uint8_t;
    using value_t = int8_t;
    using impute_t = double;
    using vec_outer_t = util::rowvec_type<outer_t>;
    using vec_inner_t = util::rowvec_type<inner_t>;
    using vec_value_t = util::rowvec_type<value_t>;
    using vec_impute_t = util::rowvec_type<impute_t>;
    using rowarr_value_t = util::rowarr_type<value_t>;
    using colarr_value_t = util::colarr_type<value_t>;
    using typename base_t::bool_t;
    using typename base_t::buffer_t;

    static constexpr size_t n_bits_per_byte = 8;
    static constexpr size_t n_categories = 3;
    static constexpr size_t chunk_size = (
        // casting helps MSVC with warning C4293
        static_cast<size_t>(1UL) << (n_bits_per_byte * sizeof(chunk_inner_t))
    );

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
    vec_outer_t _nnz;
    vec_outer_t _nnm;
    vec_impute_t _impute;
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

    outer_t cols() const { return snps(); }

    Eigen::Ref<const vec_outer_t> nnz() const 
    {
        if (!_is_read) throw_no_read();
        return _nnz;
    }

    Eigen::Ref<const vec_outer_t> nnm() const 
    {
        if (!_is_read) throw_no_read();
        return _nnm;
    }

    Eigen::Ref<const vec_impute_t> impute() const
    {
        if (!_is_read) throw_no_read();
        return _impute;
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

    const char* col_ctg(int j, size_t ctg) const
    {
        const auto _col = col(j);
        return (
            _col.data() +
            internal::read_as<outer_t>(_col.data() + sizeof(outer_t) * ctg)
        );
    }

    inner_t n_chunks(int j, size_t ctg) const
    {
        const auto* _col_ctg = col_ctg(j, ctg);
        return internal::read_as<inner_t>(_col_ctg);
    }

    iterator begin(int j, size_t ctg, size_t chnk) const
    {
        return iterator(chnk, col_ctg(j, ctg));
    }

    iterator begin(int j, size_t ctg) const
    {
        return begin(j, ctg, 0);
    }

    iterator end(int j, size_t ctg) const
    {
        return begin(j, ctg, n_chunks(j, ctg));
    }

    rowarr_value_t to_dense(
        size_t n_threads
    ) const;

    std::tuple<size_t, std::unordered_map<std::string, double>> write(
        const Eigen::Ref<const colarr_value_t>& calldata,
        const std::string& impute_method_str,
        Eigen::Ref<vec_impute_t> impute,
        size_t n_threads
    ) const;
};

} // namespace io
} // namespace adelie_core