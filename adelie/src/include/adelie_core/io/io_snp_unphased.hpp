#pragma once
#include <adelie_core/io/io_snp_base.hpp>
#include <adelie_core/io/utils.hpp>
#include <adelie_core/util/stopwatch.hpp>

namespace adelie_core {
namespace io {

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
        n_chunks(*reinterpret_cast<const inner_t*>(ctg_buffer))
    {
        if (chunk_it >= n_chunks) return;

        // increment past n_chunks and chunk_it number of chunks
        buffer_idx = sizeof(inner_t);
        for (int i = 0; i < chunk_it; ++i) {
            buffer_idx += sizeof(inner_t);
            const size_t nnz = (
                static_cast<inner_t>(*reinterpret_cast<const chunk_inner_t*>(ctg_buffer + buffer_idx))
                + 1
            );
            buffer_idx += (
                sizeof(chunk_inner_t) + nnz * sizeof(char)
            );
        }
        if (n_chunks) update();
    }

    ADELIE_CORE_STRONG_INLINE
    IOSNPChunkIterator& operator++() { 
        buffer_idx += sizeof(char);
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
        chunk_index = *reinterpret_cast<const inner_t*>(ctg_buffer + buffer_idx);
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
    using chunk_inner_t = u_char;
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
        1UL << (n_bits_per_byte * sizeof(chunk_inner_t))
    );

protected:
    static constexpr size_t _max_inner = (
        1UL << (n_bits_per_byte * sizeof(inner_t))
    );

    using base_t::throw_no_read;
    using base_t::fopen_safe;
    using base_t::is_big_endian;
    using base_t::_buffer;
    using base_t::_filename;
    using base_t::_is_read;

public:
    using iterator = IOSNPChunkIterator<
        chunk_size, inner_t, chunk_inner_t
    >;

    using base_t::base_t;
    using base_t::read;

    outer_t rows() const {
        if (!_is_read) throw_no_read();
        constexpr size_t idx = sizeof(bool_t);
        return reinterpret_cast<const outer_t&>(_buffer[idx]);
    }

    outer_t snps() const {
        if (!_is_read) throw_no_read();
        constexpr size_t idx = sizeof(bool_t) + sizeof(outer_t);
        return reinterpret_cast<const outer_t&>(_buffer[idx]);
    }

    outer_t cols() const { return snps(); }

    Eigen::Ref<const vec_outer_t> nnz() const 
    {
        if (!_is_read) throw_no_read();
        constexpr size_t slice = sizeof(bool_t) + 2 * sizeof(outer_t);
        return Eigen::Map<const vec_outer_t>(
            reinterpret_cast<const outer_t*>(&_buffer[slice]),
            cols()
        );
    }

    Eigen::Ref<const vec_outer_t> nnm() const 
    {
        if (!_is_read) throw_no_read();
        const size_t slice = sizeof(bool_t) + (2 + cols()) * sizeof(outer_t);
        return Eigen::Map<const vec_outer_t>(
            reinterpret_cast<const outer_t*>(&_buffer[slice]),
            cols()
        );
    }

    Eigen::Ref<const vec_impute_t> impute() const
    {
        if (!_is_read) throw_no_read();
        const size_t slice = sizeof(bool_t) + 2 * (1 + cols()) * sizeof(outer_t);
        return Eigen::Map<const vec_impute_t>(
            reinterpret_cast<const impute_t*>(&_buffer[slice]),
            cols()
        );
    }

    Eigen::Ref<const vec_outer_t> outer() const
    {
        if (!_is_read) throw_no_read();
        const size_t slice = sizeof(bool_t) + 2 * (1 + cols()) * sizeof(outer_t) + cols() * sizeof(impute_t);
        return Eigen::Map<const vec_outer_t>(
            reinterpret_cast<const outer_t*>(&_buffer[slice]),
            cols() + 1
        );
    }

    Eigen::Ref<const buffer_t> col(int j) const
    {
        const auto _outer = outer();
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
            reinterpret_cast<const outer_t*>(_col.data())[ctg]
        );
    }

    inner_t n_chunks(int j, size_t ctg) const
    {
        const auto* _col_ctg = col_ctg(j, ctg);
        return *reinterpret_cast<const inner_t*>(_col_ctg);
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
            for (int c = 0; c < n_categories; ++c) {
                auto it = this->begin(j, c);
                const auto end = this->end(j, c);
                const auto val = (c == 0) ? -9 : c;
                for (; it != end; ++it) {
                    dense_j[*it] = val;
                }
            }
        };
        if (n_threads <= 1) {
            for (outer_t j = 0; j < p; ++j) routine(j);
        } else {
            #pragma omp parallel for schedule(static) num_threads(n_threads)
            for (outer_t j = 0; j < p; ++j) routine(j);
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
        size_t idx = 0;
        reinterpret_cast<bool_t&>(buffer[idx]) = endian; idx += sizeof(bool_t);
        reinterpret_cast<outer_t&>(buffer[idx]) = n; idx += sizeof(outer_t);
        reinterpret_cast<outer_t&>(buffer[idx]) = p; idx += sizeof(outer_t);
        Eigen::Map<vec_outer_t>(
            reinterpret_cast<outer_t*>(&buffer[idx]),
            nnz.size()
        ) = nnz; idx += sizeof(outer_t) * nnz.size();
        Eigen::Map<vec_outer_t>(
            reinterpret_cast<outer_t*>(&buffer[idx]),
            nnm.size()
        ) = nnm; idx += sizeof(outer_t) * nnm.size();
        Eigen::Map<vec_impute_t>(
            reinterpret_cast<impute_t*>(&buffer[idx]),
            impute.size()
        ) = impute; idx += sizeof(impute_t) * impute.size();

        // outer[i] = number of bytes to jump from beginning of file 
        // to start reading column i.
        // outer[i+1] - outer[i] = total number of bytes for column i. 
        Eigen::Map<vec_outer_t> outer(
            reinterpret_cast<outer_t*>(&buffer[idx]),
            p + 1
        ); idx += sizeof(outer_t) * outer.size();
        outer[0] = idx;

        // populate outer 
        const auto outer_routine = [&](outer_t j) {
            const auto col_j = calldata.col(j);
            size_t col_bytes = 0;
            for (int i = 0; i < n_ctg; ++i) {
                col_bytes += sizeof(outer_t) + sizeof(inner_t);
                for (inner_t k = 0; k < max_chunks; ++k) {
                    const outer_t chnk = k * chunk_size;
                    bool is_nonempty = false;
                    for (inner_t c = 0; c < chunk_size; ++c) {
                        const outer_t cidx = chnk + c;
                        if (cidx >= n) break;
                        if (col_j[cidx] >= static_cast<int8_t>(n_ctg)) {
                            const auto n_ctg_str = std::to_string(n_ctg-1);
                            throw util::adelie_core_error(
                                "Detected a value greater than > " + n_ctg_str + ":"
                                "\n\tcalldata[" + std::to_string(cidx) +
                                ", " + std::to_string(j) +
                                "] = " + std::to_string(col_j[cidx]) +
                                "\nMake sure calldata only contains values <= " +
                                n_ctg_str +
                                "."
                            );
                        }
                        const bool to_not_skip = (
                            ((i == 0) && (col_j[cidx] < 0)) ||
                            ((i > 0) && (col_j[cidx] == i))
                        );
                        if (!to_not_skip) continue;
                        is_nonempty = true;
                        col_bytes += sizeof(char);
                    }
                    col_bytes += is_nonempty * (sizeof(inner_t) + sizeof(chunk_inner_t));
                }
            }
            outer[j+1] = col_bytes;
        };
        sw.start();
        if (n_threads <= 1) {
            for (outer_t j = 0; j < p; ++j) outer_routine(j);
        } else {
            #pragma omp parallel for schedule(static) num_threads(n_threads)
            for (outer_t j = 0; j < p; ++j) outer_routine(j);
        }
        benchmark["outer_time"] = sw.elapsed();

        // cumsum outer
        for (outer_t j = 0; j < p; ++j) outer[j+1] += outer[j];

        if (outer[p] > buffer.size()) {
            throw util::adelie_core_error(
                "Buffer was not initialized with a large enough size. "
                "\n\tBuffer size:   " + std::to_string(buffer.size()) +
                "\n\tExpected size: " + std::to_string(outer[p]) +
                "\nThis is likely a bug in the code. Please report it! "
            );
        }
        idx = outer[p];

        // populate (column) inner buffers
        const auto inner_routine = [&](outer_t j) {
            const auto col_j = calldata.col(j);
            Eigen::Map<buffer_t> buffer_j(
                reinterpret_cast<char*>(&buffer[outer[j]]),
                outer[j+1] - outer[j]
            );

            size_t cidx = 3 * sizeof(outer_t);

            for (int i = 0; i < n_ctg; ++i) {
                auto& outer_i = reinterpret_cast<outer_t*>(buffer_j.data())[i];
                outer_i = cidx; // IMPORTANT: relative to buffer_j not buffer!!
                auto& n_chunks = reinterpret_cast<inner_t&>(buffer_j[cidx]); 
                cidx += sizeof(inner_t);
                n_chunks = 0;

                for (inner_t k = 0; k < max_chunks; ++k) {
                    const outer_t chnk = k * chunk_size;
                    size_t curr_idx = cidx;
                    auto* chunk_index = reinterpret_cast<inner_t*>(buffer_j.data() + curr_idx); 
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
                            ((i > 0) && (col_j[didx] == i))
                        );
                        if (!to_not_skip) continue;
                        chunk_begin[nnz] = c;
                        ++nnz;
                        curr_idx += sizeof(chunk_inner_t);
                    }
                    if (nnz) {
                        *chunk_index = k;
                        *chunk_nnz = nnz - 1;
                        cidx = curr_idx;
                        ++n_chunks;
                    }
                }
            }

            if (cidx != buffer_j.size()) {
                throw util::adelie_core_error(
                    "Column index certificate does not match expected size:"
                    "\n\tCertificate:   " + std::to_string(cidx) +
                    "\n\tExpected size: " + std::to_string(buffer_j.size()) +
                    "\nThis is likely a bug in the code. Please report it! "
                );
            }
        };
        sw.start();
        if (n_threads <= 1) {
            for (outer_t j = 0; j < p; ++j) inner_routine(j);
        } else {
            #pragma omp parallel for schedule(static) num_threads(n_threads)
            for (outer_t j = 0; j < p; ++j) inner_routine(j);
        }
        benchmark["inner"] = sw.elapsed();
        
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