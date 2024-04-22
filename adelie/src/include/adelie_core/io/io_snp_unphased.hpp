#pragma once
#include <adelie_core/io/io_snp_base.hpp>
#include <adelie_core/io/utils.hpp>
#include <adelie_core/util/stopwatch.hpp>

namespace adelie_core {
namespace io {

template <size_t chunk_size, class InnerType, class ChunkInnerType>
struct IOSNPUnphasedIterator;

template <size_t chunk_size, class InnerType, class ChunkInnerType>
inline constexpr bool 
operator==(
    const IOSNPUnphasedIterator<chunk_size, InnerType, ChunkInnerType>& it1, 
    const IOSNPUnphasedIterator<chunk_size, InnerType, ChunkInnerType>& it2
)
{
    return it1.chunk_it == it2.chunk_it;
}

template <size_t chunk_size, class InnerType, class ChunkInnerType>
inline constexpr bool 
operator!=(
    const IOSNPUnphasedIterator<chunk_size, InnerType, ChunkInnerType>& it1, 
    const IOSNPUnphasedIterator<chunk_size, InnerType, ChunkInnerType>& it2
)
{
    return it1.chunk_it != it2.chunk_it;
}

template <size_t chunk_size, class InnerType, class ChunkInnerType>
struct IOSNPUnphasedIterator
{
    using inner_t = InnerType;
    using chunk_inner_t = ChunkInnerType;
    using buffer_t = util::rowvec_type<char>;
    using difference_type = inner_t;
    using value_type = inner_t;
    using pointer = value_type*;
    using reference = value_type&;
    using iterator_category = std::forward_iterator_tag;
    using iterator = IOSNPUnphasedIterator;

    inner_t chunk_it;
    const Eigen::Map<const buffer_t> col_buffer;
    const inner_t n_chunks;
    size_t buffer_idx = 0;
    inner_t chunk_index;
    inner_t chunk_nnz;
    inner_t inner;
    inner_t dense_chunk_index;
    inner_t dense_index;

    IOSNPUnphasedIterator(
        inner_t chunk_it,
        const Eigen::Ref<const buffer_t>& col_buffer
    ):
        chunk_it(chunk_it),
        col_buffer(col_buffer.data(), col_buffer.size()),
        n_chunks(*reinterpret_cast<const inner_t*>(col_buffer.data()))
    {
        buffer_idx += sizeof(inner_t);
        if (n_chunks) update();
    }

    iterator& operator++() { 
        ++buffer_idx;
        ++inner;
        if (inner >= chunk_nnz) {
            ++chunk_it;
            if (chunk_it < n_chunks) update();
        } else {
            dense_index = (
                dense_chunk_index +
                *reinterpret_cast<const chunk_inner_t*>(col_buffer.data() + buffer_idx)
            );
        }
        return *this; 
    }
    reference operator*() { return dense_index; }
    friend constexpr bool operator==<>(const IOSNPUnphasedIterator&, 
                                       const IOSNPUnphasedIterator&);
    friend constexpr bool operator!=<>(const IOSNPUnphasedIterator&, 
                                       const IOSNPUnphasedIterator&);

    ADELIE_CORE_STRONG_INLINE
    void update()
    {
        chunk_index = *reinterpret_cast<const inner_t*>(col_buffer.data() + buffer_idx);
        buffer_idx += sizeof(inner_t);
        chunk_nnz = (
            static_cast<inner_t>(*reinterpret_cast<const chunk_inner_t*>(col_buffer.data() + buffer_idx))
            + 1
        );
        buffer_idx += sizeof(chunk_inner_t);
        inner = 0;
        dense_chunk_index = chunk_index * chunk_size;
        dense_index = (
            dense_chunk_index +
            *reinterpret_cast<const chunk_inner_t*>(col_buffer.data() + buffer_idx)
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

protected:
    static constexpr size_t _n_bits_per_byte = 8;
    static constexpr size_t _n_categories = 3;
    static constexpr size_t _chunk_size = 1 << (_n_bits_per_byte * sizeof(chunk_inner_t));
    static constexpr size_t _multiplier = (
        sizeof(inner_t) + 
        sizeof(value_t)
    );

    using base_t::throw_no_read;
    using base_t::fopen_safe;
    using base_t::is_big_endian;
    using base_t::_buffer;
    using base_t::_filename;
    using base_t::_is_read;

public:
    using iterator = IOSNPUnphasedIterator<_chunk_size, inner_t, chunk_inner_t>;

    using base_t::base_t;
    using base_t::read;

    inner_t rows() const {
        if (!_is_read) throw_no_read();
        return reinterpret_cast<const inner_t&>(_buffer[sizeof(bool_t)]);
    }

    inner_t snps() const {
        if (!_is_read) throw_no_read();
        return reinterpret_cast<const inner_t&>(_buffer[sizeof(bool_t) + sizeof(inner_t)]);
    }

    inner_t cols() const { return snps(); }

    Eigen::Ref<const vec_inner_t> nnz() const 
    {
        if (!_is_read) throw_no_read();
        constexpr size_t slice = sizeof(bool_t) + 2 * sizeof(inner_t);
        return Eigen::Map<const vec_inner_t>(
            reinterpret_cast<const inner_t*>(&_buffer[slice]),
            cols()
        );
    }

    Eigen::Ref<const vec_inner_t> nnm() const 
    {
        if (!_is_read) throw_no_read();
        const size_t slice = sizeof(bool_t) + (2 + cols()) * sizeof(inner_t);
        return Eigen::Map<const vec_inner_t>(
            reinterpret_cast<const inner_t*>(&_buffer[slice]),
            cols()
        );
    }

    Eigen::Ref<const vec_impute_t> impute() const
    {
        if (!_is_read) throw_no_read();
        const size_t slice = sizeof(bool_t) + 2 * (1 + cols()) * sizeof(inner_t);
        return Eigen::Map<const vec_impute_t>(
            reinterpret_cast<const impute_t*>(&_buffer[slice]),
            cols()
        );
    }

    Eigen::Ref<const buffer_t> col(size_t ctg, int j) const
    {
        if (!_is_read) throw_no_read();
        if (ctg >= _n_categories) {
            throw util::adelie_core_error(
                "Category must be 0, 1, or 2."
            );
        }
        const size_t slice = sizeof(bool_t) + 2 * (1 + cols()) * sizeof(inner_t) + cols() * sizeof(impute_t);
        Eigen::Map<const vec_outer_t> ctg_outer(
            reinterpret_cast<const outer_t*>(&_buffer[slice]),
            _n_categories + 1
        );
        Eigen::Map<const vec_outer_t> outer(
            reinterpret_cast<const outer_t*>(&_buffer[ctg_outer[ctg]]),
            cols() + 1
        );
        Eigen::Map<const buffer_t> buffer_ctg_j(
            reinterpret_cast<const char*>(&_buffer[outer[j]]),
            outer[j+1] - outer[j]
        );
        return buffer_ctg_j;
    }

    iterator begin(size_t ctg, int j) const
    {
        return iterator(0, col(ctg, j));
    }

    iterator end(size_t ctg, int j) const
    {
        const auto _col = col(ctg, j);
        const auto n_chunks = *reinterpret_cast<const inner_t*>(_col.data());
        return iterator(n_chunks, _col);
    }

    rowarr_value_t to_dense(
        size_t n_threads
    ) const
    {
        if (!_is_read) throw_no_read();
        const auto n = rows();
        const auto p = cols();
        rowarr_value_t dense(n, p);

        const auto routine = [&](inner_t j) {
            auto dense_j = dense.col(j);
            dense_j.setZero();
            util::rowvec_type<value_t, 3> fills;
            fills[0] = -9;
            fills[1] = 1;
            fills[2] = 2;
            for (int i = 0; i < 3; ++i)
            {
                const auto fill_i = fills[i];
                const auto _end = end(i, j);
                auto it = begin(i, j);
                for (; it != _end; ++it) {
                    dense_j[*it] = fill_i;
                }
            }
        };
        if (n_threads <= 1) {
            for (inner_t j = 0; j < p; ++j) routine(j);
        } else {
            #pragma omp parallel for schedule(auto) num_threads(n_threads)
            for (inner_t j = 0; j < p; ++j) routine(j);
        }

        return dense;
    }

    std::tuple<size_t, std::unordered_map<std::string, double>> write(
        const Eigen::Ref<const colarr_value_t>& calldata,
        const std::string& impute_method_str,
        Eigen::Ref<vec_impute_t> impute,
        size_t n_threads
    )
    {
        using sw_t = util::Stopwatch;

        sw_t sw;
        std::unordered_map<std::string, double> benchmark;

        const bool_t endian = is_big_endian();
        const inner_t n = calldata.rows();
        const inner_t p = calldata.cols();

        // handle impute_method
        const auto impute_method = util::convert_impute_method(impute_method_str);

        // compute impute
        sw.start();
        compute_impute(calldata, impute_method, impute, n_threads);
        benchmark["impute"] = sw.elapsed();

        // compute number of non-missing values
        vec_inner_t nnm(p);
        sw.start();
        compute_nnm(calldata, impute_method, nnm, n_threads);
        benchmark["nnm"] = sw.elapsed();

        // compute number of non-zero values
        vec_inner_t nnz(p);
        sw.start();
        compute_nnz(calldata, impute_method, nnz, n_threads);
        benchmark["nnz"] = sw.elapsed();

        // allocate sufficient memory (upper bound on size)
        constexpr size_t n_ctg = _n_categories;
        const size_t preamble_size = (
            sizeof(bool_t) +                    // endian
            2 * sizeof(inner_t) +               // n, p
            nnz.size() * sizeof(inner_t) +      // nnz
            nnm.size() * sizeof(inner_t) +      // nnm
            impute.size() * sizeof(impute_t) +  // impute
            (n_ctg + 1) * sizeof(outer_t)       // outer (category)
        );
        const size_t max_chunks = (n + _chunk_size - 1) / _chunk_size;
        buffer_t buffer(
            preamble_size +
            n_ctg * (p + 1) * (                 // for each category and column
                sizeof(outer_t) +               // outer[j]
                sizeof(inner_t) +               // n_chunks
                max_chunks * (                  // max_chunks * (chunk-outer + chunk-inner)
                    sizeof(inner_t) + 
                    sizeof(chunk_inner_t)
                )
            ) +
            nnz.sum() * sizeof(chunk_inner_t)   // nnz * char
        );

        // populate buffer
        size_t idx = 0;
        reinterpret_cast<bool_t&>(buffer[idx]) = endian; idx += sizeof(bool_t);
        reinterpret_cast<inner_t&>(buffer[idx]) = n; idx += sizeof(inner_t);
        reinterpret_cast<inner_t&>(buffer[idx]) = p; idx += sizeof(inner_t);
        Eigen::Map<vec_inner_t>(
            reinterpret_cast<inner_t*>(&buffer[idx]),
            nnz.size()
        ) = nnz; idx += sizeof(inner_t) * nnz.size();
        Eigen::Map<vec_inner_t>(
            reinterpret_cast<inner_t*>(&buffer[idx]),
            nnm.size()
        ) = nnm; idx += sizeof(inner_t) * nnm.size();
        Eigen::Map<vec_impute_t>(
            reinterpret_cast<impute_t*>(&buffer[idx]),
            impute.size()
        ) = impute; idx += sizeof(impute_t) * impute.size();

        // ctg_outer[i] = number of bytes to jump from beginning of file 
        // to start reading category i.
        // ctg_outer[i+1] - ctg_outer[i] = total number of bytes for category i. 
        // Possible categories:
        //  - 0: negative value (NA)
        //  - 1/2: 1/2 value.
        Eigen::Map<vec_outer_t> ctg_outer(
            reinterpret_cast<outer_t*>(&buffer[idx]),
            n_ctg + 1
        ); idx += sizeof(outer_t) * ctg_outer.size();
        ctg_outer[0] = idx;

        double outer_time = 0;
        double inner_time = 0;
        
        for (int i = 0; i < ctg_outer.size()-1; ++i) {
            // outer vector across columns
            Eigen::Map<vec_outer_t> outer(
                reinterpret_cast<outer_t*>(&buffer[idx]),
                p+1
            ); idx += sizeof(outer_t) * outer.size();
            outer[0] = idx;

            // populate outer 
            const auto outer_routine = [&](inner_t j) {
                const auto col_j = calldata.col(j);
                size_t col_bytes = sizeof(inner_t);
                for (inner_t k = 0; k < max_chunks; ++k) {
                    const inner_t chnk = k * _chunk_size;
                    bool is_nonempty = false;
                    for (inner_t c = 0; c < _chunk_size; ++c) {
                        const inner_t idx = chnk + c;
                        if (idx >= n) break;
                        if (col_j[idx] >= static_cast<int8_t>(n_ctg)) {
                            const auto n_ctg_str = std::to_string(n_ctg-1);
                            throw util::adelie_core_error(
                                "Detected a value greater than > " +
                                n_ctg_str +
                                "! Make sure the matrix only contains values <= " +
                                n_ctg_str +
                                "."
                            );
                        }
                        const bool to_not_skip = (
                            ((i == 0) && (col_j[idx] < 0)) ||
                            ((i > 0) && (col_j[idx] == i))
                        );
                        if (!to_not_skip) continue;
                        is_nonempty = true;
                        ++col_bytes;
                    }
                    col_bytes += is_nonempty * (sizeof(inner_t) + sizeof(chunk_inner_t));
                } 
                outer[j+1] = col_bytes;
            };
            sw.start();
            if (n_threads <= 1) {
                for (inner_t j = 0; j < p; ++j) outer_routine(j);
            } else {
                #pragma omp parallel for schedule(auto) num_threads(n_threads)
                for (inner_t j = 0; j < p; ++j) outer_routine(j);
            }
            outer_time += sw.elapsed();

            for (inner_t j = 0; j < p; ++j) outer[j+1] += outer[j];

            idx = outer[p];

            // populate next ctg_outer
            ctg_outer[i+1] = idx;

            // populate column buffers
            const auto routine = [&](inner_t j) {
                const auto col_j = calldata.col(j);
                Eigen::Map<buffer_t> buffer_j(
                    reinterpret_cast<char*>(&buffer[outer[j]]),
                    outer[j+1] - outer[j]
                );

                size_t cidx = 0;
                auto& n_chunks = reinterpret_cast<inner_t&>(buffer_j[cidx]); cidx += sizeof(inner_t);
                n_chunks = 0;
                for (inner_t k = 0; k < max_chunks; ++k) {
                    const inner_t chnk = k * _chunk_size;
                    size_t curr_idx = cidx;
                    auto* chunk_index = reinterpret_cast<inner_t*>(buffer_j.data() + curr_idx); curr_idx += sizeof(inner_t);
                    auto* chunk_nnz = reinterpret_cast<chunk_inner_t*>(buffer_j.data() + curr_idx); curr_idx += sizeof(chunk_inner_t);
                    auto* chunk_begin = reinterpret_cast<chunk_inner_t*>(buffer_j.data() + curr_idx); 
                    inner_t nnz = 0;
                    for (inner_t c = 0; c < _chunk_size; ++c) {
                        const inner_t didx = chnk + c;
                        if (didx >= n) break;
                        const bool to_not_skip = (
                            ((i == 0) && (col_j[didx] < 0)) ||
                            ((i > 0) && (col_j[didx] == i))
                        );
                        if (!to_not_skip) continue;
                        chunk_begin[nnz] = c;
                        ++nnz;
                        ++curr_idx;
                    }
                    if (nnz) {
                        *chunk_index = k;
                        *chunk_nnz = nnz - 1;
                        cidx = curr_idx;
                        ++n_chunks;
                    }
                }

                if (cidx != buffer_j.size()) {
                    throw util::adelie_core_error(
                        "Column index certificate does not match expected size. "
                        "This is likely a bug in the code. Please report it! "
                    );
                }
            };

            sw.start();
            if (n_threads <= 1) {
                for (inner_t j = 0; j < p; ++j) routine(j);
            } else {
                #pragma omp parallel for schedule(auto) num_threads(n_threads)
                for (inner_t j = 0; j < p; ++j) routine(j);
            }
            inner_time += sw.elapsed();
        }

        if (idx > buffer.size()) {
            throw util::adelie_core_error(
                "Buffer was not initialized with a large enough size. "
                "This is likely a bug in the code. Please report it! "
            );
        }

        benchmark["outer"] = outer_time;
        benchmark["inner"] = inner_time;

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