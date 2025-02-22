#pragma once
#include <adelie_core/io/io_snp_unphased.hpp>
#include <adelie_core/io/utils.hpp>
#include <adelie_core/util/omp.hpp>
#include <adelie_core/util/stopwatch.hpp>

namespace adelie_core {
namespace io {

ADELIE_CORE_IO_SNP_UNPHASED_TP
size_t
ADELIE_CORE_IO_SNP_UNPHASED::read() 
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

ADELIE_CORE_IO_SNP_UNPHASED_TP
typename ADELIE_CORE_IO_SNP_UNPHASED::rowarr_value_t
ADELIE_CORE_IO_SNP_UNPHASED::to_dense(
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
    util::omp_parallel_for(routine, 0, p, n_threads);

    return dense;
}

ADELIE_CORE_IO_SNP_UNPHASED_TP
std::tuple<size_t, std::unordered_map<std::string, double>>
ADELIE_CORE_IO_SNP_UNPHASED::write(
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
    util::omp_parallel_for(outer_routine, 0, p, n_threads);
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
    util::omp_parallel_for(inner_routine, 0, p, n_threads);
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

} // namespace io
} // namespace adelie_core