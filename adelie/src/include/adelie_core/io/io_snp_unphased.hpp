#pragma once
#include <adelie_core/io/io_snp_base.hpp>
#include <adelie_core/io/utils.hpp>
#include <adelie_core/util/stopwatch.hpp>

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

    IOSNPChunkIterator(
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

    size_t read() override
    {
        const size_t total_bytes = base_t::read();

        size_t idx = sizeof(bool_t);

        _rows = internal::read_as<outer_t>(_buffer.data() + idx);
        idx += sizeof(outer_t);

        _snps = internal::read_as<outer_t>(_buffer.data() + idx);
        idx += sizeof(outer_t);

        _nnz.resize(_snps);
        std::memcpy(_nnz.data(), _buffer.data() + idx, sizeof(outer_t) * _snps);
        idx += sizeof(outer_t) * _snps;

        _nnm.resize(_snps);
        std::memcpy(_nnm.data(), _buffer.data() + idx, sizeof(outer_t) * _snps);
        idx += sizeof(outer_t) * _snps;

        _impute.resize(_snps);
        std::memcpy(_impute.data(), _buffer.data() + idx, sizeof(impute_t) * _snps);
        idx += sizeof(impute_t) * _snps;

        _outer.resize(_snps + 1);
        std::memcpy(_outer.data(), _buffer.data() + idx, sizeof(outer_t) * (_snps + 1));
        idx += sizeof(outer_t) * (_snps + 1);

        return total_bytes;
    }

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
    ) const
    {
        const auto n = rows();
        const auto p = cols();
        rowarr_value_t dense(n, p);

        const auto routine = [&](outer_t j) {
            auto dense_j = dense.col(j);
            dense_j.setZero();
            for (size_t c = 0; c < n_categories; ++c) {
                auto it = this->begin(j, c);
                const auto end = this->end(j, c);
                const int val = (c == 0) ? -9 : c;
                for (; it != end; ++it) {
                    dense_j[*it] = val;
                }
            }
        };
        if (n_threads <= 1) {
            for (int j = 0; j < static_cast<int>(p); ++j) routine(j);
        } else {
            #pragma omp parallel for schedule(static) num_threads(n_threads)
            for (int j = 0; j < static_cast<int>(p); ++j) routine(j);
        }

        return dense;
    }

    std::tuple<size_t, std::unordered_map<std::string, double>> write(
        const Eigen::Ref<const colarr_value_t>& calldata,
        const std::string& impute_method_str,
        Eigen::Ref<vec_impute_t> impute,
        size_t n_threads
    ) const
    {
        using sw_t = util::Stopwatch;

        sw_t sw;
        std::unordered_map<std::string, double> benchmark;

        const bool_t endian = is_big_endian();
        const outer_t n = calldata.rows();
        const outer_t p = calldata.cols();

        const size_t max_chunks = (n + chunk_size - 1) / chunk_size;
        if (max_chunks >= _max_inner) {
            throw util::adelie_core_error(
                "calldata dimensions are too large! "
            );
        } 

        // handle impute_method
        const auto impute_method = util::convert_impute_method(impute_method_str);

        // compute impute
        sw.start();
        compute_impute(calldata, impute_method, impute, n_threads);
        benchmark["impute"] = sw.elapsed();

        // compute number of non-missing values
        vec_outer_t nnm(p);
        sw.start();
        compute_nnm(calldata, nnm, n_threads);
        benchmark["nnm"] = sw.elapsed();

        // compute number of non-zero values
        vec_outer_t nnz(p);
        sw.start();
        compute_nnz(calldata, nnz, n_threads);
        benchmark["nnz"] = sw.elapsed();

        // allocate sufficient memory (upper bound on size)
        constexpr size_t n_ctg = n_categories; // alias
        const size_t preamble_size = (
            sizeof(bool_t) +                    // endian
            2 * sizeof(outer_t) +               // n, p
            nnz.size() * sizeof(outer_t) +      // nnz
            nnm.size() * sizeof(outer_t) +      // nnm
            impute.size() * sizeof(impute_t) +  // impute
            (p + 1) * sizeof(outer_t)           // outer (columns)
        );
        buffer_t buffer(
            preamble_size +
            p * n_ctg * (           // for each snp, category
                sizeof(outer_t) +       // outer (category)
                sizeof(inner_t) +       // n_chunks
                max_chunks * (            // for each chunk
                    sizeof(inner_t) +       // chunk index
                    sizeof(chunk_inner_t)   // chunk nnz - 1
                )
            ) +
            nnz.sum() * sizeof(chunk_inner_t)   // nnz * char
        );

        // populate buffer
        outer_t idx = 0;
        std::memcpy(buffer.data()+idx, &endian, sizeof(bool_t)); idx += sizeof(bool_t);
        std::memcpy(buffer.data()+idx, &n, sizeof(outer_t)); idx += sizeof(outer_t);
        std::memcpy(buffer.data()+idx, &p, sizeof(outer_t)); idx += sizeof(outer_t);
        std::memcpy(buffer.data()+idx, nnz.data(), sizeof(outer_t) * nnz.size());
        idx += sizeof(outer_t) * nnz.size();
        std::memcpy(buffer.data()+idx, nnm.data(), sizeof(outer_t) * nnm.size());
        idx += sizeof(outer_t) * nnm.size();
        std::memcpy(buffer.data()+idx, impute.data(), sizeof(impute_t) * impute.size());
        idx += sizeof(impute_t) * impute.size();

        // outer[i] = number of bytes to jump from beginning of file 
        // to start reading column i.
        // outer[i+1] - outer[i] = total number of bytes for column i. 
        char* const outer_ptr = buffer.data() + idx;
        const size_t outer_size = p + 1;
        idx += sizeof(outer_t) * outer_size;
        std::memcpy(outer_ptr, &idx, sizeof(outer_t));

        // flag to detect any errors
        std::atomic_bool try_failed = false;

        // populate outer 
        const auto outer_routine = [&](outer_t j) {
            if (try_failed.load(std::memory_order_relaxed)) return;

            const auto col_j = calldata.col(j);
            outer_t col_bytes = 0;
            for (size_t i = 0; i < n_ctg; ++i) {
                col_bytes += sizeof(outer_t) + sizeof(inner_t);
                for (inner_t k = 0; k < max_chunks; ++k) {
                    const outer_t chnk = k * chunk_size;
                    bool is_nonempty = false;
                    for (inner_t c = 0; c < chunk_size; ++c) {
                        const outer_t cidx = chnk + c;
                        if (cidx >= n) break;
                        if (col_j[cidx] >= static_cast<int8_t>(n_ctg)) {
                            try_failed = true;
                            return;
                        }
                        const bool to_not_skip = (
                            ((i == 0) && (col_j[cidx] < 0)) ||
                            ((i > 0) && (col_j[cidx] == static_cast<char>(i)))
                        );
                        if (!to_not_skip) continue;
                        is_nonempty = true;
                        col_bytes += sizeof(char);
                    }
                    col_bytes += is_nonempty * (sizeof(inner_t) + sizeof(chunk_inner_t));
                }
            }
            std::memcpy(outer_ptr + sizeof(outer_t) * (j+1), &col_bytes, sizeof(outer_t));
        };
        sw.start();
        if (n_threads <= 1) {
            for (int j = 0; j < static_cast<int>(p); ++j) outer_routine(j);
        } else {
            #pragma omp parallel for schedule(static) num_threads(n_threads)
            for (int j = 0; j < static_cast<int>(p); ++j) outer_routine(j);
        }
        benchmark["outer_time"] = sw.elapsed();
        
        if (try_failed) {
            const auto n_ctg_str = std::to_string(n_ctg-1);
            throw util::adelie_core_error(
                "Detected a value greater than > " + n_ctg_str + ". "
                "Make sure calldata only contains values <= " + n_ctg_str + ". "
            );
        }

        // cumsum outer
        for (outer_t j = 0; j < p; ++j) {
            const outer_t outer_curr = internal::read_as<outer_t>(outer_ptr + sizeof(outer_t) * (j+1));
            const outer_t outer_prev = internal::read_as<outer_t>(outer_ptr + sizeof(outer_t) * j);
            const outer_t sum = outer_curr + outer_prev; 
            std::memcpy(outer_ptr + sizeof(outer_t) * (j+1), &sum, sizeof(outer_t));
        }

        const outer_t outer_last = internal::read_as<outer_t>(outer_ptr + sizeof(outer_t) * p);
        if (outer_last > static_cast<size_t>(buffer.size())) {
            throw util::adelie_core_error(
                "Buffer was not initialized with a large enough size. "
                "\n\tBuffer size:   " + std::to_string(buffer.size()) +
                "\n\tExpected size: " + std::to_string(outer_last) +
                "\nThis is likely a bug in the code. Please report it! "
            );
        }
        idx = outer_last;

        // populate (column) inner buffers
        try_failed = false;
        const auto inner_routine = [&](outer_t j) {
            if (try_failed.load(std::memory_order_relaxed)) return;

            const auto col_j = calldata.col(j);
            const outer_t outer_curr = internal::read_as<outer_t>(outer_ptr + sizeof(outer_t) * (j+1));
            const outer_t outer_prev = internal::read_as<outer_t>(outer_ptr + sizeof(outer_t) * j);
            Eigen::Map<buffer_t> buffer_j(
                buffer.data() + outer_prev,
                outer_curr - outer_prev
            );

            outer_t cidx = 3 * sizeof(outer_t);

            for (size_t i = 0; i < n_ctg; ++i) {
                // IMPORTANT: relative to buffer_j not buffer!!
                std::memcpy(buffer_j.data() + sizeof(outer_t) * i, &cidx, sizeof(outer_t));
                auto* n_chunks_ptr = buffer_j.data() + cidx;
                cidx += sizeof(inner_t);
                inner_t n_chunks = 0;

                for (inner_t k = 0; k < max_chunks; ++k) {
                    const outer_t chnk = k * chunk_size;
                    size_t curr_idx = cidx;
                    auto* chunk_index = buffer_j.data() + curr_idx; 
                    curr_idx += sizeof(inner_t);
                    auto* chunk_nnz = reinterpret_cast<chunk_inner_t*>(buffer_j.data() + curr_idx); 
                    curr_idx += sizeof(chunk_inner_t);
                    auto* chunk_begin = reinterpret_cast<chunk_inner_t*>(buffer_j.data() + curr_idx); 
                    inner_t nnz = 0;
                    for (inner_t c = 0; c < chunk_size; ++c) {
                        const outer_t didx = chnk + c;
                        if (didx >= n) break;
                        const bool to_not_skip = (
                            ((i == 0) && (col_j[didx] < 0)) ||
                            ((i > 0) && (col_j[didx] == static_cast<char>(i)))
                        );
                        if (!to_not_skip) continue;
                        chunk_begin[nnz] = c;
                        ++nnz;
                        curr_idx += sizeof(chunk_inner_t);
                    }
                    if (nnz) {
                        std::memcpy(chunk_index, &k, sizeof(inner_t));
                        *chunk_nnz = nnz - 1;
                        cidx = curr_idx;
                        ++n_chunks;
                    }
                }
                std::memcpy(n_chunks_ptr, &n_chunks, sizeof(inner_t));
            }

            if (cidx != static_cast<size_t>(buffer_j.size())) {
                try_failed = true;
            }
        };
        sw.start();
        if (n_threads <= 1) {
            for (int j = 0; j < static_cast<int>(p); ++j) inner_routine(j);
        } else {
            #pragma omp parallel for schedule(static) num_threads(n_threads)
            for (int j = 0; j < static_cast<int>(p); ++j) inner_routine(j);
        }
        benchmark["inner"] = sw.elapsed();

        if (try_failed) {
            throw util::adelie_core_error(
                "Column index certificate does not match expected size. "
                "This is likely a bug in the code. Please report it! "
            );
        }
        
        sw.start();
        auto file_ptr = fopen_safe(_filename.c_str(), "wb");
        auto fp = file_ptr.get();
        auto total_bytes = std::fwrite(buffer.data(), sizeof(char), idx, fp);
        if (total_bytes != static_cast<size_t>(idx)) {
            throw util::adelie_core_error(
                "Could not write the full buffer."
            );
        }
        benchmark["write"] = sw.elapsed();

        return {total_bytes, benchmark};
    }
};

} // namespace io
} // namespace adelie_core