#pragma once
#include <adelie_core/io/io_snp_combine_r.hpp>
#include <adelie_core/io/utils.hpp>
#include <adelie_core/util/omp.hpp>
#include <adelie_core/util/stopwatch.hpp>

namespace adelie_core {
namespace io {

ADELIE_CORE_IO_SNP_COMBINE_R_TP
size_t
ADELIE_CORE_IO_SNP_COMBINE_R::read() 
{
    const size_t total_bytes = base_t::read();

    size_t idx = sizeof(bool_t);

    _rows = internal::read_as<outer_t>(_buffer.data() + idx);
    idx += sizeof(outer_t);

    _snps = internal::read_as<outer_t>(_buffer.data() + idx);
    idx += sizeof(outer_t);

    _ancestries = internal::read_as<chunk_inner_t>(_buffer.data() + idx);
    idx += sizeof(chunk_inner_t);

    _cols = _snps * (1 + _ancestries);

    _nnz.resize(_cols);
    std::memcpy(_nnz.data(), _buffer.data() + idx, sizeof(outer_t) * _cols);
    idx += sizeof(outer_t) * _cols;

    _outer.resize(_snps + 1);
    std::memcpy(_outer.data(), _buffer.data() + idx, sizeof(outer_t) * (_snps + 1));
    idx += sizeof(outer_t) * (_snps + 1);

    return total_bytes;
}

ADELIE_CORE_IO_SNP_COMBINE_R_TP
typename ADELIE_CORE_IO_SNP_COMBINE_R::rowarr_value_t
ADELIE_CORE_IO_SNP_COMBINE_R::to_dense(
    size_t n_threads
) const
{
    const size_t n = rows();
    const size_t s = snps();
    const size_t A = ancestries();
    rowarr_value_t dense(n, s * (1 + A));

    const auto routine = [&](outer_t j) {
        auto dense_j = dense.middleCols((1 + A) * j, 1 + A);
        dense_j.setZero();
        
        // First column: SNP data
        auto it = this->begin(j, 0);
        const auto end = this->end(j, 0);
        for (; it != end; ++it) {
            dense_j(*it, 0) += 1;
        }

        // Next A columns: Ancestry dosages
        for (size_t a = 0; a < A; ++a) {
            auto it = this->begin(j, a + 1);
            const auto end = this->end(j, a + 1);
            for (; it != end; ++it) {
                dense_j(*it, a + 1) += 1;
            }
        }
    };
    util::omp_parallel_for(routine, 0, s, n_threads);

    return dense;
}

ADELIE_CORE_IO_SNP_COMBINE_R_TP
std::tuple<size_t, std::unordered_map<std::string, double>>
ADELIE_CORE_IO_SNP_COMBINE_R::write(
    const Eigen::Ref<const colarr_value_t>& calldata,
    const Eigen::Ref<const colarr_value_t>& ancestries,
    size_t A,
    size_t n_threads
) const
{
    using sw_t = util::Stopwatch;

    if (
        (calldata.rows() != ancestries.rows()) ||
        (2 * calldata.cols() != ancestries.cols())
    ) {
        throw util::adelie_core_error(
            "ancestries must have shape (n, 2*s) where calldata has shape (n,s)."
        );
    }

    if (A >= chunk_size) {
        throw util::adelie_core_error(
            "Number of ancestries A must be < " +
            std::to_string(chunk_size) +
            "."
        );
    }

    sw_t sw;
    std::unordered_map<std::string, double> benchmark;

    const bool_t endian = is_big_endian();
    const outer_t n = calldata.rows();
    const outer_t s = calldata.cols();

    const size_t max_chunks = (n + chunk_size - 1) / chunk_size;
    if (max_chunks >= _max_inner) {
        throw util::adelie_core_error(
            "calldata dimensions are too large! "
        );
    } 

    // ------------------------------------------------------------------
    // 1.  Compute sample_nnz (for file header) and payload_nnz separately
    // ------------------------------------------------------------------
    vec_outer_t sample_nnz(s * (1 + A));
    vec_outer_t payload_nnz(s * (1 + A));
    sample_nnz.setZero();
    payload_nnz.setZero();
    sw.start();

    // — SNP data column —
    // sample_nnz = count of rows where calldata != 0
    // payload_nnz = sum of calldata values (0,1,2)
    for (size_t j = 0; j < s; ++j) {
        const auto cal_j = calldata.col(j);
        outer_t samples = 0, bytes = 0;
        for (outer_t i = 0; i < n; ++i) {
            auto v = cal_j.coeff(i);
            if (v != 0) ++samples;
            bytes += static_cast<outer_t>(v);
        }
        size_t col_idx = (1 + A) * j;
        sample_nnz[col_idx]   = samples;
        payload_nnz[col_idx]  = bytes;
    }

    // — Ancestry dosage columns —
    // sample_nnz = count of rows where at least one haplo==a
    // payload_nnz = count of *all* matching haplotypes (0–2)
    for (size_t a = 0; a < A; ++a) {
        for (size_t j = 0; j < s; ++j) {
            const auto anc0 = ancestries.col(2*j);
            const auto anc1 = ancestries.col(2*j+1);
            outer_t samples = 0, bytes = 0;
            for (outer_t i = 0; i < n; ++i) {
                bool hit0 = (anc0.coeff(i) == static_cast<char>(a));
                bool hit1 = (anc1.coeff(i) == static_cast<char>(a));
                if (hit0 || hit1) ++samples;
                if (hit0) ++bytes;
                if (hit1) ++bytes;
            }
            size_t col_idx = (1 + A) * j + (a + 1);
            sample_nnz[col_idx]  = samples;
            payload_nnz[col_idx] = bytes;
        }
    }
    benchmark["nnz"] = sw.elapsed();



    // allocate sufficient memory (upper bound on size)
    const size_t preamble_size = (
        sizeof(bool_t) +                    // endian
        2 * sizeof(outer_t) +               // n, s
        sizeof(chunk_inner_t) +             // A
        payload_nnz.size() * sizeof(outer_t) +     // payload_nnz
        (s + 1) * sizeof(outer_t)           // outer (snps)
    );
    buffer_t buffer(
        preamble_size +
        // for each SNP, reserve space for its (1 + A) columns structure:
        s * (
            (1 + A) * sizeof(outer_t) +     // outer pointers for each column
            (1 + A) * (
                sizeof(inner_t) +               // n_chunks
                max_chunks * (                  // for each chunk
                    sizeof(inner_t) +               // chunk index
                    sizeof(chunk_inner_t)           // chunk nnz - 1
                )
            )
        ) +
        payload_nnz.sum() * sizeof(chunk_inner_t)   // actual data entries
    );

    // populate buffer
    outer_t idx = 0;
    std::memcpy(buffer.data()+idx, &endian, sizeof(bool_t)); idx += sizeof(bool_t);
    std::memcpy(buffer.data()+idx, &n, sizeof(outer_t)); idx += sizeof(outer_t);
    std::memcpy(buffer.data()+idx, &s, sizeof(outer_t)); idx += sizeof(outer_t);
    std::memcpy(buffer.data()+idx, &A, sizeof(chunk_inner_t)); idx += sizeof(chunk_inner_t);
    std::memcpy(buffer.data()+idx, sample_nnz.data(), sizeof(outer_t) * sample_nnz.size());
    idx += sizeof(outer_t) * sample_nnz.size();

    // outer[i] = number of bytes to jump from beginning of file 
    // to start reading snp i.
    // outer[i+1] - outer[i] = total number of bytes for snp i. 
    char* const outer_ptr = buffer.data() + idx;
    const size_t outer_size = s + 1;
    idx += sizeof(outer_t) * outer_size;
    std::memcpy(outer_ptr, &idx, sizeof(outer_t));

    // flag to detect any errors
    std::atomic_char try_failed = 0;

    // populate outer 
    const auto outer_routine = [&](outer_t j) {
        if (try_failed.load(std::memory_order_relaxed)) return;

        // Start with the space for (1+A) outer pointers
        outer_t snp_bytes = (1 + A) * sizeof(outer_t);

        // --- SNP data column (col 0) ---
        snp_bytes += sizeof(inner_t);              // n_chunks
        const auto cal_j = calldata.col(j);
        for (inner_t k = 0; k < max_chunks; ++k) {
            const outer_t base = k * chunk_size;
            // count how many bytes we'll emit in this chunk
            inner_t nnz = 0;
            for (inner_t c = 0; c < chunk_size; ++c) {
                const outer_t didx = base + c;
                if (didx >= n) break;
                // each non‐zero allele produces one byte
                nnz += cal_j[didx];
            }
            // chunk header (index + nnz-1) if nonempty
            if (nnz > 0) {
                snp_bytes += sizeof(inner_t)     // chunk index
                        + sizeof(chunk_inner_t) // chunk_nnz
                        + nnz * sizeof(chunk_inner_t);
            }
        }

        // --- Ancestry dosage columns (cols 1…A) ---
        const auto anc_j0 = ancestries.col(2*j);
        const auto anc_j1 = ancestries.col(2*j + 1);
        for (size_t a = 0; a < A; ++a) {
            snp_bytes += sizeof(inner_t);        // n_chunks for this ancestry
            for (inner_t k = 0; k < max_chunks; ++k) {
                const outer_t base = k * chunk_size;
                inner_t nnz = 0;
                for (inner_t c = 0; c < chunk_size; ++c) {
                    const outer_t didx = base + c;
                    if (didx >= n) break;
                    // one entry per haplotype
                    if (anc_j0[didx] == static_cast<char>(a)) ++nnz;
                    if (anc_j1[didx] == static_cast<char>(a)) ++nnz;
                }
                if (nnz > 0) {
                    snp_bytes += sizeof(inner_t)     // chunk index
                            + sizeof(chunk_inner_t) // chunk_nnz
                            + nnz * sizeof(chunk_inner_t);
                }
            }
        }

        // write the per-SNP byte count into the outer table
        std::memcpy(
        outer_ptr + sizeof(outer_t) * (j+1),
        &snp_bytes,
        sizeof(outer_t)
        );
    };
    sw.start();
    util::omp_parallel_for(outer_routine, 0, s, n_threads);
    benchmark["outer_time"] = sw.elapsed();

    switch (try_failed) {
        case 1: {
            throw util::adelie_core_error(
                "Detected an ancestry not in the range [0, A). "
                "Make sure ancestries only contains values in [0, A). "
            );
            break;
        }
        case 2: {
            throw util::adelie_core_error(
                "Detected a value not in [0,2]. "
                "Make sure calldata only contains values in [0,2]. "
            );
            break;
        }
    }

    // cumsum outer
    for (outer_t j = 0; j < s; ++j) {
        const outer_t outer_curr = internal::read_as<outer_t>(outer_ptr + sizeof(outer_t) * (j+1));
        const outer_t outer_prev = internal::read_as<outer_t>(outer_ptr + sizeof(outer_t) * j);
        const outer_t sum = outer_curr + outer_prev; 
        std::memcpy(outer_ptr + sizeof(outer_t) * (j+1), &sum, sizeof(outer_t));
    }

    const outer_t outer_last = internal::read_as<outer_t>(outer_ptr + sizeof(outer_t) * s);

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
    try_failed = 0;
    const auto inner_routine = [&](outer_t j) {
        if (try_failed.load(std::memory_order_relaxed)) return;

        // 1) Slice out this SNP's block
        const outer_t outer_curr = internal::read_as<outer_t>(
            outer_ptr + sizeof(outer_t)*(j+1)
        );
        const outer_t outer_prev = internal::read_as<outer_t>(
            outer_ptr + sizeof(outer_t)*j
        );
        Eigen::Map<buffer_t> buffer_j(
            buffer.data() + outer_prev,
            outer_curr - outer_prev
        );

        // 2) Skip over the (1+A) outer pointers
        outer_t cidx = (1 + A) * sizeof(outer_t);

        //
        // --- SNP data column (col 0 of the block) ---
        //
        // write pointer to our chunk data
        std::memcpy(buffer_j.data() + 0*sizeof(outer_t), &cidx, sizeof(outer_t));
        // reserve space for 'n_chunks'
        auto *n_chunks_ptr0 = reinterpret_cast<inner_t*>(buffer_j.data() + cidx);
        cidx += sizeof(inner_t);
        inner_t n_chunks0 = 0;

        // grab the whole column once
        const auto cal_j = calldata.col(j);

        // for each chunk
        for (inner_t k = 0; k < max_chunks; ++k) {
            const outer_t base = k * chunk_size;
            size_t curr_idx = cidx;

            // chunk header
            auto *chunk_index = reinterpret_cast<inner_t*>(buffer_j.data() + curr_idx);
            curr_idx += sizeof(inner_t);
            auto *chunk_nnz   = reinterpret_cast<chunk_inner_t*>(
                buffer_j.data() + curr_idx
            );
            curr_idx += sizeof(chunk_inner_t);

            // now list out positions, *once per allele*
            auto *chunk_begin = reinterpret_cast<chunk_inner_t*>(
                buffer_j.data() + curr_idx
            );
            inner_t nnz = 0;
            for (inner_t c = 0; c < chunk_size; ++c) {
                const outer_t didx = base + c;
                if (didx >= n) break;
                // if cal_j[didx] is 1 or 2, emit that many entries
                for (int rep = 0; rep < cal_j[didx]; ++rep) {
                    chunk_begin[nnz++] = c;
                    curr_idx += sizeof(chunk_inner_t);
                }
            }

            if (nnz) {
                // commit header and advance
                std::memcpy(chunk_index, &k, sizeof(inner_t));
                *chunk_nnz = nnz - 1;
                cidx = curr_idx;
                ++n_chunks0;
            }
        }
        // write back total chunk‐count
        std::memcpy(n_chunks_ptr0, &n_chunks0, sizeof(inner_t));


        //
        // --- Ancestry dosage columns (cols 1…A of the block) ---
        //
        // grab both haplotype‐columns once
        const auto anc_j0 = ancestries.col(2*j);
        const auto anc_j1 = ancestries.col(2*j + 1);

        for (size_t a = 0; a < A; ++a) {
            // pointer for this ancestry col
            std::memcpy(
                buffer_j.data() + (1 + a)*sizeof(outer_t),
                &cidx,
                sizeof(outer_t)
            );
            auto *n_chunks_ptr = reinterpret_cast<inner_t*>(
                buffer_j.data() + cidx
            );
            cidx += sizeof(inner_t);
            inner_t n_chunks = 0;

            for (inner_t k = 0; k < max_chunks; ++k) {
                const outer_t base = k * chunk_size;
                size_t curr_idx = cidx;

                auto *chunk_index = reinterpret_cast<inner_t*>(
                    buffer_j.data() + curr_idx
                );
                curr_idx += sizeof(inner_t);
                auto *chunk_nnz = reinterpret_cast<chunk_inner_t*>(
                    buffer_j.data() + curr_idx
                );
                curr_idx += sizeof(chunk_inner_t);

                auto *chunk_begin = reinterpret_cast<chunk_inner_t*>(
                    buffer_j.data() + curr_idx
                );
                inner_t nnz = 0;

                // emit one entry for each haplotype of ancestry 'a'
                for (inner_t c = 0; c < chunk_size; ++c) {
                    const outer_t didx = base + c;
                    if (didx >= n) break;
                    if (anc_j0[didx] == static_cast<char>(a)) {
                        chunk_begin[nnz++] = c;
                        curr_idx += sizeof(chunk_inner_t);
                    }
                    if (anc_j1[didx] == static_cast<char>(a)) {
                        chunk_begin[nnz++] = c;
                        curr_idx += sizeof(chunk_inner_t);
                    }
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

        // final size check
        if (cidx != static_cast<outer_t>(buffer_j.size())) {
            try_failed = 1;
        }
    };

    sw.start();
    util::omp_parallel_for(inner_routine, 0, s, n_threads);
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

// ----------------------------------------------------------------------------
// Overload: write with selected ancestry list
// ----------------------------------------------------------------------------
#include <algorithm>

namespace adelie_core {
namespace io {

ADELIE_CORE_IO_SNP_COMBINE_R_TP
std::tuple<size_t, std::unordered_map<std::string, double>>
ADELIE_CORE_IO_SNP_COMBINE_R::write(
    const Eigen::Ref<const colarr_value_t>& calldata,
    const Eigen::Ref<const colarr_value_t>& ancestries,
    size_t A_total,
    const std::vector<uint32_t>& selected_ancestries,
    size_t n_threads
) const
{
    using sw_t = util::Stopwatch;

    if (
        (calldata.rows() != ancestries.rows()) ||
        (2 * calldata.cols() != ancestries.cols())
    ) {
        throw util::adelie_core_error(
            "ancestries must have shape (n, 2*s) where calldata has shape (n,s)."
        );
    }

    // Validate selection
    if (selected_ancestries.empty()) {
        throw util::adelie_core_error(
            "selected_ancestries must be non-empty."
        );
    }
    for (auto a : selected_ancestries) {
        if (a >= A_total) {
            throw util::adelie_core_error(
                "selected_ancestries contains an index out of range [0, A_total)."
            );
        }
    }

    // Ensure entries are unique while preserving the user-provided order
    std::vector<uint32_t> sel;
    sel.reserve(selected_ancestries.size());
    for (auto a : selected_ancestries) {
        if (std::find(sel.begin(), sel.end(), a) == sel.end()) {
            sel.push_back(a);
        }
    }

    // Now proceed similarly to the A-based writer, with A := sel.size() and
    // ancestry references restricted to 'sel'. Keep A_total only for validation.

    const bool_t endian = is_big_endian();
    const outer_t n = calldata.rows();
    const outer_t s = calldata.cols();
    const size_t A_sel = sel.size();

    if (A_sel >= chunk_size) {
        throw util::adelie_core_error(
            "Number of selected ancestries must be < " +
            std::to_string(chunk_size) +
            "."
        );
    }

    const size_t max_chunks = (n + chunk_size - 1) / chunk_size;
    if (max_chunks >= _max_inner) {
        throw util::adelie_core_error(
            "calldata dimensions are too large! "
        );
    }

    sw_t sw;
    std::unordered_map<std::string, double> benchmark;

    // nnz counts for header (samples with nonzero per column of output)
    vec_outer_t sample_nnz(s * (1 + A_sel));
    vec_outer_t payload_nnz(s * (1 + A_sel));
    sample_nnz.setZero();
    payload_nnz.setZero();

    sw.start();
    // SNP data column stats
    for (size_t j = 0; j < s; ++j) {
        const auto cal_j = calldata.col(j);
        outer_t samples = 0, bytes = 0;
        for (outer_t i = 0; i < n; ++i) {
            auto v = cal_j.coeff(i);
            if (v != 0) ++samples;
            bytes += static_cast<outer_t>(v);
        }
        size_t col_idx = (1 + A_sel) * j;
        sample_nnz[col_idx]   = samples;
        payload_nnz[col_idx]  = bytes;
    }

    // Ancestry dosage columns stats (restricted to sel)
    for (size_t ai = 0; ai < A_sel; ++ai) {
        const auto a = sel[ai];
        for (size_t j = 0; j < s; ++j) {
            const auto anc0 = ancestries.col(2*j);
            const auto anc1 = ancestries.col(2*j+1);
            outer_t samples = 0, bytes = 0;
            for (outer_t i = 0; i < n; ++i) {
                bool hit0 = (anc0.coeff(i) == static_cast<char>(a));
                bool hit1 = (anc1.coeff(i) == static_cast<char>(a));
                if (hit0 || hit1) ++samples;
                if (hit0) ++bytes;
                if (hit1) ++bytes;
            }
            size_t col_idx = (1 + A_sel) * j + (ai + 1);
            sample_nnz[col_idx]  = samples;
            payload_nnz[col_idx] = bytes;
        }
    }
    benchmark["nnz"] = sw.elapsed();

    // Buffer pre-allocation
    const size_t preamble_size = (
        sizeof(bool_t) +                    // endian
        2 * sizeof(outer_t) +               // n, s
        sizeof(chunk_inner_t) +             // A_sel (encoded as chunk_inner_t)
        sample_nnz.size() * sizeof(outer_t) +     // nnz per output column
        (s + 1) * sizeof(outer_t)           // outer (snps)
    );
    buffer_t buffer(
        preamble_size +
        s * (
            (1 + A_sel) * sizeof(outer_t) +
            (1 + A_sel) * (
                sizeof(inner_t) +
                max_chunks * (
                    sizeof(inner_t) +
                    sizeof(chunk_inner_t)
                )
            )
        ) +
        payload_nnz.sum() * sizeof(chunk_inner_t)
    );

    // Populate header
    outer_t idx = 0;
    std::memcpy(buffer.data()+idx, &endian, sizeof(bool_t)); idx += sizeof(bool_t);
    std::memcpy(buffer.data()+idx, &n, sizeof(outer_t)); idx += sizeof(outer_t);
    std::memcpy(buffer.data()+idx, &s, sizeof(outer_t)); idx += sizeof(outer_t);
    const auto A_sel_byte = static_cast<chunk_inner_t>(A_sel);
    std::memcpy(buffer.data()+idx, &A_sel_byte, sizeof(chunk_inner_t)); idx += sizeof(chunk_inner_t);
    std::memcpy(buffer.data()+idx, sample_nnz.data(), sizeof(outer_t) * sample_nnz.size());
    idx += sizeof(outer_t) * sample_nnz.size();

    // outer table
    char* const outer_ptr = buffer.data() + idx;
    const size_t outer_size = s + 1;
    idx += sizeof(outer_t) * outer_size;
    std::memcpy(outer_ptr, &idx, sizeof(outer_t));

    std::atomic_char try_failed = 0;

    // Compute per-SNP byte sizes
    const auto outer_routine = [&](outer_t j) {
        if (try_failed.load(std::memory_order_relaxed)) return;

        outer_t snp_bytes = (1 + A_sel) * sizeof(outer_t);

        // Genotype column
        snp_bytes += sizeof(inner_t);
        const auto cal_j = calldata.col(j);
        for (inner_t k = 0; k < max_chunks; ++k) {
            const outer_t base = k * chunk_size;
            inner_t nnz = 0;
            for (inner_t c = 0; c < chunk_size; ++c) {
                const outer_t didx = base + c;
                if (didx >= n) break;
                nnz += cal_j[didx];
            }
            if (nnz > 0) {
                snp_bytes += sizeof(inner_t) + sizeof(chunk_inner_t) + nnz * sizeof(chunk_inner_t);
            }
        }

        // Selected ancestry columns
        const auto anc_j0 = ancestries.col(2*j);
        const auto anc_j1 = ancestries.col(2*j + 1);
        for (size_t ai = 0; ai < A_sel; ++ai) {
            const auto a = sel[ai];
            snp_bytes += sizeof(inner_t);
            for (inner_t k = 0; k < max_chunks; ++k) {
                const outer_t base = k * chunk_size;
                inner_t nnz = 0;
                for (inner_t c = 0; c < chunk_size; ++c) {
                    const outer_t didx = base + c;
                    if (didx >= n) break;
                    if (anc_j0[didx] == static_cast<char>(a)) ++nnz;
                    if (anc_j1[didx] == static_cast<char>(a)) ++nnz;
                }
                if (nnz > 0) {
                    snp_bytes += sizeof(inner_t) + sizeof(chunk_inner_t) + nnz * sizeof(chunk_inner_t);
                }
            }
        }

        std::memcpy(
            outer_ptr + sizeof(outer_t) * (j+1),
            &snp_bytes,
            sizeof(outer_t)
        );
    };
    sw.start();
    util::omp_parallel_for(outer_routine, 0, s, n_threads);
    benchmark["outer_time"] = sw.elapsed();

    // cumsum outer
    for (outer_t j = 0; j < s; ++j) {
        const outer_t outer_curr = internal::read_as<outer_t>(outer_ptr + sizeof(outer_t) * (j+1));
        const outer_t outer_prev = internal::read_as<outer_t>(outer_ptr + sizeof(outer_t) * j);
        const outer_t sum = outer_curr + outer_prev;
        std::memcpy(outer_ptr + sizeof(outer_t) * (j+1), &sum, sizeof(outer_t));
    }

    const outer_t outer_last = internal::read_as<outer_t>(outer_ptr + sizeof(outer_t) * s);
    if (outer_last > static_cast<size_t>(buffer.size())) {
        throw util::adelie_core_error(
            "Buffer was not initialized with a large enough size. "
        );
    }
    idx = outer_last;

    // Fill per-SNP blocks
    const auto inner_routine = [&](outer_t j) {
        if (try_failed.load(std::memory_order_relaxed)) return;

        const outer_t outer_curr = internal::read_as<outer_t>(outer_ptr + sizeof(outer_t) * (j+1));
        const outer_t outer_prev = internal::read_as<outer_t>(outer_ptr + sizeof(outer_t) * j);
        Eigen::Map<buffer_t> buffer_j(
            buffer.data() + outer_prev,
            outer_curr - outer_prev
        );

        outer_t cidx = (1 + A_sel) * sizeof(outer_t);

        // Genotype column pointer
        std::memcpy(buffer_j.data() + 0*sizeof(outer_t), &cidx, sizeof(outer_t));
        auto *n_chunks_ptr0 = reinterpret_cast<inner_t*>(buffer_j.data() + cidx);
        cidx += sizeof(inner_t);
        inner_t n_chunks0 = 0;

        const auto cal_j = calldata.col(j);
        for (inner_t k = 0; k < max_chunks; ++k) {
            const outer_t base = k * chunk_size;
            size_t curr_idx = cidx;
            auto *chunk_index = reinterpret_cast<inner_t*>(buffer_j.data() + curr_idx); curr_idx += sizeof(inner_t);
            auto *chunk_nnz   = reinterpret_cast<chunk_inner_t*>(buffer_j.data() + curr_idx); curr_idx += sizeof(chunk_inner_t);
            auto *chunk_begin = reinterpret_cast<chunk_inner_t*>(buffer_j.data() + curr_idx);
            inner_t nnz = 0;
            for (inner_t c = 0; c < chunk_size; ++c) {
                const outer_t didx = base + c;
                if (didx >= n) break;
                for (int rep = 0; rep < cal_j[didx]; ++rep) {
                    chunk_begin[nnz++] = c;
                    curr_idx += sizeof(chunk_inner_t);
                }
            }
            if (nnz) {
                std::memcpy(chunk_index, &k, sizeof(inner_t));
                *chunk_nnz = nnz - 1;
                cidx = curr_idx;
                ++n_chunks0;
            }
        }
        std::memcpy(n_chunks_ptr0, &n_chunks0, sizeof(inner_t));

        // Selected ancestry columns
        const auto anc_j0 = ancestries.col(2*j);
        const auto anc_j1 = ancestries.col(2*j + 1);
        for (size_t ai = 0; ai < A_sel; ++ai) {
            const auto a = sel[ai];
            std::memcpy(buffer_j.data() + (1 + ai)*sizeof(outer_t), &cidx, sizeof(outer_t));
            auto *n_chunks_ptr = reinterpret_cast<inner_t*>(buffer_j.data() + cidx);
            cidx += sizeof(inner_t);
            inner_t n_chunks = 0;
            for (inner_t k = 0; k < max_chunks; ++k) {
                const outer_t base = k * chunk_size;
                size_t curr_idx = cidx;
                auto *chunk_index = reinterpret_cast<inner_t*>(buffer_j.data() + curr_idx); curr_idx += sizeof(inner_t);
                auto *chunk_nnz   = reinterpret_cast<chunk_inner_t*>(buffer_j.data() + curr_idx); curr_idx += sizeof(chunk_inner_t);
                auto *chunk_begin = reinterpret_cast<chunk_inner_t*>(buffer_j.data() + curr_idx);
                inner_t nnz = 0;
                for (inner_t c = 0; c < chunk_size; ++c) {
                    const outer_t didx = base + c;
                    if (didx >= n) break;
                    if (anc_j0[didx] == static_cast<char>(a)) { chunk_begin[nnz++] = c; curr_idx += sizeof(chunk_inner_t); }
                    if (anc_j1[didx] == static_cast<char>(a)) { chunk_begin[nnz++] = c; curr_idx += sizeof(chunk_inner_t); }
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

        if (cidx != static_cast<outer_t>(buffer_j.size())) {
            try_failed = 1;
        }
    };
    sw.start();
    util::omp_parallel_for(inner_routine, 0, s, n_threads);
    benchmark["inner"] = sw.elapsed();

    if (try_failed) {
        throw util::adelie_core_error(
            "Column index certificate does not match expected size. "
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