#pragma once
#include <adelie_core/io/io_snp_combine_s.hpp>
#include <adelie_core/io/utils.hpp>
#include <adelie_core/util/omp.hpp>
#include <adelie_core/util/stopwatch.hpp>
#include <algorithm>

namespace adelie_core {
namespace io {

ADELIE_CORE_IO_SNP_COMBINE_S_TP
size_t
ADELIE_CORE_IO_SNP_COMBINE_S::read()
{
    const size_t total_bytes = base_t::read();

    size_t idx = sizeof(bool_t);

    _rows = internal::read_as<outer_t>(_buffer.data() + idx);
    idx += sizeof(outer_t);

    _snps = internal::read_as<outer_t>(_buffer.data() + idx);
    idx += sizeof(outer_t);

    _ancestries = internal::read_as<chunk_inner_t>(_buffer.data() + idx);
    idx += sizeof(chunk_inner_t);

    _cols = _snps * (2 * _ancestries);

    _nnz.resize(_cols);
    std::memcpy(_nnz.data(), _buffer.data() + idx, sizeof(outer_t) * _cols);
    idx += sizeof(outer_t) * _cols;

    _outer.resize(_snps + 1);
    std::memcpy(_outer.data(), _buffer.data() + idx, sizeof(outer_t) * (_snps + 1));
    idx += sizeof(outer_t) * (_snps + 1);

    return total_bytes;
}

ADELIE_CORE_IO_SNP_COMBINE_S_TP
typename ADELIE_CORE_IO_SNP_COMBINE_S::rowarr_value_t
ADELIE_CORE_IO_SNP_COMBINE_S::to_dense(
    size_t n_threads
) const
{
    const size_t n = rows();
    const size_t s = snps();
    const size_t A = ancestries();
    rowarr_value_t dense(n, s * (2 * A));

    const auto routine = [&](outer_t j) {
        auto dense_j = dense.middleCols((2 * A) * j, 2 * A);
        dense_j.setZero();

        // First A columns: mutated haplotype counts per ancestry
        for (size_t a = 0; a < A; ++a) {
            auto it = this->begin(j, static_cast<int>(a));
            const auto end = this->end(j, static_cast<int>(a));
            for (; it != end; ++it) {
                dense_j(*it, a) += 1;
            }
        }

        // Next A columns: ancestry dosage counts per ancestry (like unphased ancestry dosage)
        for (size_t a = 0; a < A; ++a) {
            auto it = this->begin(j, static_cast<int>(A + a));
            const auto end = this->end(j, static_cast<int>(A + a));
            for (; it != end; ++it) {
                dense_j(*it, A + a) += 1;
            }
        }
    };
    util::omp_parallel_for(routine, 0, s, n_threads);

    return dense;
}

ADELIE_CORE_IO_SNP_COMBINE_S_TP
std::tuple<size_t, std::unordered_map<std::string, double>>
ADELIE_CORE_IO_SNP_COMBINE_S::write(
    const Eigen::Ref<const colarr_value_t>& calldata,
    const Eigen::Ref<const colarr_value_t>& ancestries,
    size_t A,
    size_t n_threads
) const
{
    using sw_t = util::Stopwatch;

    // calldata and ancestries are phased: shape (n, 2*s)
    if (
        (calldata.rows() != ancestries.rows()) ||
        (calldata.cols() != ancestries.cols()) ||
        (calldata.cols() % 2)
    ) {
        throw util::adelie_core_error(
            "calldata and ancestries must have shape (n, 2*s)."
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
    const outer_t s = calldata.cols() / 2;

    const size_t max_chunks = (n + chunk_size - 1) / chunk_size;
    if (max_chunks >= _max_inner) {
        throw util::adelie_core_error(
            "calldata dimensions are too large! "
        );
    }

    // nnz per output column (2*A per SNP)
    vec_outer_t nnz(s * (2 * A));
    nnz.setZero();
    sw.start();
    // First A: mutated hap counts (sum over haps)
    for (size_t a = 0; a < A; ++a) {
        for (size_t j = 0; j < s; ++j) {
            const auto cal0 = calldata.col(2*j);
            const auto cal1 = calldata.col(2*j+1);
            const auto anc0 = ancestries.col(2*j);
            const auto anc1 = ancestries.col(2*j+1);
            outer_t samples = 0;
            for (outer_t i = 0; i < n; ++i) {
                bool contribute = (
                    (cal0[i] == 1 && anc0[i] == static_cast<char>(a)) ||
                    (cal1[i] == 1 && anc1[i] == static_cast<char>(a))
                );
                if (contribute) ++samples; // sample nnz counts rows with any hit
            }
            nnz[(2*A)*j + a] = samples;
        }
    }
    // Next A: ancestry dosage counts
    for (size_t a = 0; a < A; ++a) {
        for (size_t j = 0; j < s; ++j) {
            const auto anc0 = ancestries.col(2*j);
            const auto anc1 = ancestries.col(2*j+1);
            outer_t samples = 0;
            for (outer_t i = 0; i < n; ++i) {
                if ((anc0[i] == static_cast<char>(a)) || (anc1[i] == static_cast<char>(a))) ++samples;
            }
            nnz[(2*A)*j + (A + a)] = samples;
        }
    }
    benchmark["nnz"] = sw.elapsed();

    // Pre-allocate buffer
    const size_t preamble_size = (
        sizeof(bool_t) +
        2 * sizeof(outer_t) +
        sizeof(chunk_inner_t) +
        nnz.size() * sizeof(outer_t) +
        (s + 1) * sizeof(outer_t)
    );
    buffer_t buffer(
        preamble_size +
        s * (
            (2*A) * sizeof(outer_t) +
            (2*A) * (
                sizeof(inner_t) +
                max_chunks * (
                    sizeof(inner_t) +
                    sizeof(chunk_inner_t)
                )
            )
        ) +
        // payload: number of actual entries written per column equals
        // mutated hap counts (each hit writes 1) and dosage counts (each hap hit writes 1)
        // Use nnz as an upper bound for sample nnz; worst-case payload could be larger for dosage.
        // To keep it simple and safe, over-allocate by factor 2.
        2 * nnz.sum() * sizeof(chunk_inner_t)
    );

    // Header
    outer_t idx = 0;
    std::memcpy(buffer.data()+idx, &endian, sizeof(bool_t)); idx += sizeof(bool_t);
    std::memcpy(buffer.data()+idx, &n, sizeof(outer_t)); idx += sizeof(outer_t);
    std::memcpy(buffer.data()+idx, &s, sizeof(outer_t)); idx += sizeof(outer_t);
    std::memcpy(buffer.data()+idx, &A, sizeof(chunk_inner_t)); idx += sizeof(chunk_inner_t);
    std::memcpy(buffer.data()+idx, nnz.data(), sizeof(outer_t) * nnz.size());
    idx += sizeof(outer_t) * nnz.size();

    // outer table (per SNP block)
    char* const outer_ptr = buffer.data() + idx;
    idx += sizeof(outer_t) * (s + 1);
    std::memcpy(outer_ptr, &idx, sizeof(outer_t));

    std::atomic_char try_failed = 0;

    // Outer routine: compute per-SNP byte sizes
    const auto outer_routine = [&](outer_t j) {
        if (try_failed.load(std::memory_order_relaxed)) return;
        outer_t snp_bytes = (2*A) * sizeof(outer_t);

        const auto cal0 = calldata.col(2*j);
        const auto cal1 = calldata.col(2*j+1);
        const auto anc0 = ancestries.col(2*j);
        const auto anc1 = ancestries.col(2*j+1);

        // First A mutated-hap columns (write one entry per mutated hap)
        for (size_t a = 0; a < A; ++a) {
            snp_bytes += sizeof(inner_t);
            for (inner_t k = 0; k < max_chunks; ++k) {
                const outer_t base = k * chunk_size;
                inner_t nnz_chunk = 0;
                for (inner_t c = 0; c < chunk_size; ++c) {
                    const outer_t didx = base + c;
                    if (didx >= n) break;
                    // Validate
                    if ((cal0[didx] != 0 && cal0[didx] != 1) || (cal1[didx] != 0 && cal1[didx] != 1)) { try_failed = 2; return; }
                    if (anc0[didx] < 0 || anc0[didx] >= static_cast<char>(A) || anc1[didx] < 0 || anc1[didx] >= static_cast<char>(A)) { try_failed = 1; return; }
                    if (cal0[didx] == 1 && anc0[didx] == static_cast<char>(a)) ++nnz_chunk;
                    if (cal1[didx] == 1 && anc1[didx] == static_cast<char>(a)) ++nnz_chunk;
                }
                if (nnz_chunk > 0) {
                    snp_bytes += sizeof(inner_t) + sizeof(chunk_inner_t) + nnz_chunk * sizeof(chunk_inner_t);
                }
            }
        }

        // Next A dosage columns
        for (size_t a = 0; a < A; ++a) {
            snp_bytes += sizeof(inner_t);
            for (inner_t k = 0; k < max_chunks; ++k) {
                const outer_t base = k * chunk_size;
                inner_t nnz_chunk = 0;
                for (inner_t c = 0; c < chunk_size; ++c) {
                    const outer_t didx = base + c;
                    if (didx >= n) break;
                    if (anc0[didx] < 0 || anc0[didx] >= static_cast<char>(A) || anc1[didx] < 0 || anc1[didx] >= static_cast<char>(A)) { try_failed = 1; return; }
                    if (anc0[didx] == static_cast<char>(a)) ++nnz_chunk;
                    if (anc1[didx] == static_cast<char>(a)) ++nnz_chunk;
                }
                if (nnz_chunk > 0) {
                    snp_bytes += sizeof(inner_t) + sizeof(chunk_inner_t) + nnz_chunk * sizeof(chunk_inner_t);
                }
            }
        }

        std::memcpy(outer_ptr + sizeof(outer_t) * (j+1), &snp_bytes, sizeof(outer_t));
    };
    util::omp_parallel_for(outer_routine, 0, s, n_threads);
    switch (try_failed) {
        case 1: {
            throw util::adelie_core_error(
                "Detected an ancestry not in the range [0, A). Make sure ancestries only contains values in [0, A). "
            );
        }
        case 2: {
            throw util::adelie_core_error(
                "Detected a non-binary value. Make sure calldata only contains 0 or 1 values. "
            );
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
        throw util::adelie_core_error("Buffer was not initialized with a large enough size. ");
    }
    idx = outer_last;

    // Inner routine: write per-SNP block detail
    const auto inner_routine = [&](outer_t j) {
        if (try_failed.load(std::memory_order_relaxed)) return;

        const outer_t outer_curr = internal::read_as<outer_t>(outer_ptr + sizeof(outer_t) * (j+1));
        const outer_t outer_prev = internal::read_as<outer_t>(outer_ptr + sizeof(outer_t) * j);
        Eigen::Map<buffer_t> buffer_j(
            buffer.data() + outer_prev,
            outer_curr - outer_prev
        );

        outer_t cidx = (2*A) * sizeof(outer_t);

        const auto cal0 = calldata.col(2*j);
        const auto cal1 = calldata.col(2*j+1);
        const auto anc0 = ancestries.col(2*j);
        const auto anc1 = ancestries.col(2*j+1);

        // First A: mutated hap columns (emit one entry per mutated hap)
        for (size_t a = 0; a < A; ++a) {
            std::memcpy(buffer_j.data() + sizeof(outer_t) * a, &cidx, sizeof(outer_t));
            auto *n_chunks_ptr = reinterpret_cast<inner_t*>(buffer_j.data() + cidx);
            cidx += sizeof(inner_t);
            inner_t n_chunks = 0;
            for (inner_t k = 0; k < max_chunks; ++k) {
                const outer_t base = k * chunk_size;
                size_t curr_idx = cidx;
                auto *chunk_index = reinterpret_cast<inner_t*>(buffer_j.data() + curr_idx); curr_idx += sizeof(inner_t);
                auto *chunk_nnz   = reinterpret_cast<chunk_inner_t*>(buffer_j.data() + curr_idx); curr_idx += sizeof(chunk_inner_t);
                auto *chunk_begin = reinterpret_cast<chunk_inner_t*>(buffer_j.data() + curr_idx);
                inner_t nnz_chunk = 0;
                for (inner_t c = 0; c < chunk_size; ++c) {
                    const outer_t didx = base + c;
                    if (didx >= n) break;
                    if ((cal0[didx] != 0 && cal0[didx] != 1) || (cal1[didx] != 0 && cal1[didx] != 1)) { try_failed = 2; return; }
                    if (anc0[didx] < 0 || anc0[didx] >= static_cast<char>(A) || anc1[didx] < 0 || anc1[didx] >= static_cast<char>(A)) { try_failed = 1; return; }
                    if (cal0[didx] == 1 && anc0[didx] == static_cast<char>(a)) { chunk_begin[nnz_chunk++] = c; curr_idx += sizeof(chunk_inner_t); }
                    if (cal1[didx] == 1 && anc1[didx] == static_cast<char>(a)) { chunk_begin[nnz_chunk++] = c; curr_idx += sizeof(chunk_inner_t); }
                }
                if (nnz_chunk) { std::memcpy(chunk_index, &k, sizeof(inner_t)); *chunk_nnz = nnz_chunk - 1; cidx = curr_idx; ++n_chunks; }
            }
            std::memcpy(n_chunks_ptr, &n_chunks, sizeof(inner_t));
        }

        // Next A: ancestry dosage columns
        for (size_t a = 0; a < A; ++a) {
            std::memcpy(buffer_j.data() + sizeof(outer_t) * (A + a), &cidx, sizeof(outer_t));
            auto *n_chunks_ptr = reinterpret_cast<inner_t*>(buffer_j.data() + cidx);
            cidx += sizeof(inner_t);
            inner_t n_chunks = 0;
            for (inner_t k = 0; k < max_chunks; ++k) {
                const outer_t base = k * chunk_size;
                size_t curr_idx = cidx;
                auto *chunk_index = reinterpret_cast<inner_t*>(buffer_j.data() + curr_idx); curr_idx += sizeof(inner_t);
                auto *chunk_nnz   = reinterpret_cast<chunk_inner_t*>(buffer_j.data() + curr_idx); curr_idx += sizeof(chunk_inner_t);
                auto *chunk_begin = reinterpret_cast<chunk_inner_t*>(buffer_j.data() + curr_idx);
                inner_t nnz_chunk = 0;
                for (inner_t c = 0; c < chunk_size; ++c) {
                    const outer_t didx = base + c;
                    if (didx >= n) break;
                    if (anc0[didx] == static_cast<char>(a)) { chunk_begin[nnz_chunk++] = c; curr_idx += sizeof(chunk_inner_t); }
                    if (anc1[didx] == static_cast<char>(a)) { chunk_begin[nnz_chunk++] = c; curr_idx += sizeof(chunk_inner_t); }
                }
                if (nnz_chunk) { std::memcpy(chunk_index, &k, sizeof(inner_t)); *chunk_nnz = nnz_chunk - 1; cidx = curr_idx; ++n_chunks; }
            }
            std::memcpy(n_chunks_ptr, &n_chunks, sizeof(inner_t));
        }

        if (cidx != static_cast<outer_t>(buffer_j.size())) { try_failed = 3; }
    };
    util::omp_parallel_for(inner_routine, 0, s, n_threads);

    switch (try_failed) {
        case 1: throw util::adelie_core_error("Detected an ancestry not in the range [0, A). ");
        case 2: throw util::adelie_core_error("Detected a non-binary value. ");
        case 3: throw util::adelie_core_error("Column index certificate does not match expected size. ");
    }

    auto file_ptr = fopen_safe(_filename.c_str(), "wb");
    auto fp = file_ptr.get();
    auto total_bytes = std::fwrite(buffer.data(), sizeof(char), idx, fp);
    if (total_bytes != static_cast<size_t>(idx)) {
        throw util::adelie_core_error("Could not write the full buffer.");
    }
    benchmark["write"] = 0.0; // minimal tracking here

    return {total_bytes, benchmark};
}

ADELIE_CORE_IO_SNP_COMBINE_S_TP
std::tuple<size_t, std::unordered_map<std::string, double>>
ADELIE_CORE_IO_SNP_COMBINE_S::write(
    const Eigen::Ref<const colarr_value_t>& calldata,
    const Eigen::Ref<const colarr_value_t>& ancestries,
    size_t A_total,
    const std::vector<uint32_t>& selected_ancestries,
    size_t n_threads
) const
{
    // Implement by filtering ancestries and delegating to the non-selected write
    if (selected_ancestries.empty()) {
        throw util::adelie_core_error("selected_ancestries must be non-empty.");
    }
    for (auto a : selected_ancestries) {
        if (a >= A_total) {
            throw util::adelie_core_error("selected_ancestries contains an index out of range [0, A_total).");
        }
    }
    // Ensure uniqueness preserving order
    std::vector<uint32_t> sel;
    sel.reserve(selected_ancestries.size());
    for (auto a : selected_ancestries) {
        if (std::find(sel.begin(), sel.end(), a) == sel.end()) sel.push_back(a);
    }

    const outer_t n = calldata.rows();
    const outer_t s = calldata.cols() / 2;
    const size_t B = sel.size();

    // Build temporary dense then serialize via the dense path: reuse the same writer logic by constructing columns directly
    // For efficiency, we implement selected-version by emitting directly, mirroring the unselected write with A := B and ancestry index remapped via sel.

    using sw_t = util::Stopwatch;
    sw_t sw;
    std::unordered_map<std::string, double> benchmark;

    const bool_t endian = is_big_endian();
    const size_t max_chunks = (n + chunk_size - 1) / chunk_size;
    if (max_chunks >= _max_inner) {
        throw util::adelie_core_error("calldata dimensions are too large! ");
    }

    vec_outer_t nnz(s * (2 * B)); nnz.setZero();
    // nnz for mutated-hap columns
    for (size_t bi = 0; bi < B; ++bi) {
        const auto a = sel[bi];
        for (size_t j = 0; j < s; ++j) {
            const auto cal0 = calldata.col(2*j);
            const auto cal1 = calldata.col(2*j+1);
            const auto anc0 = ancestries.col(2*j);
            const auto anc1 = ancestries.col(2*j+1);
            outer_t samples = 0;
            for (outer_t i = 0; i < n; ++i) {
                bool contribute = (
                    (cal0[i] == 1 && anc0[i] == static_cast<char>(a)) ||
                    (cal1[i] == 1 && anc1[i] == static_cast<char>(a))
                );
                if (contribute) ++samples;
            }
            nnz[(2*B)*j + bi] = samples;
        }
    }
    // nnz for dosage columns
    for (size_t bi = 0; bi < B; ++bi) {
        const auto a = sel[bi];
        for (size_t j = 0; j < s; ++j) {
            const auto anc0 = ancestries.col(2*j);
            const auto anc1 = ancestries.col(2*j+1);
            outer_t samples = 0;
            for (outer_t i = 0; i < n; ++i) {
                if ((anc0[i] == static_cast<char>(a)) || (anc1[i] == static_cast<char>(a))) ++samples;
            }
            nnz[(2*B)*j + (B + bi)] = samples;
        }
    }

    const outer_t n_cols = s * (2 * B);
    const size_t preamble_size = (
        sizeof(bool_t) + 2*sizeof(outer_t) + sizeof(chunk_inner_t) + n_cols*sizeof(outer_t) + (s+1)*sizeof(outer_t)
    );
    buffer_t buffer(
        preamble_size +
        s * (
            (2*B) * sizeof(outer_t) +
            (2*B) * ( sizeof(inner_t) + max_chunks * (sizeof(inner_t) + sizeof(chunk_inner_t)) )
        ) + 2 * nnz.sum() * sizeof(chunk_inner_t)
    );

    outer_t idx = 0;
    std::memcpy(buffer.data()+idx, &endian, sizeof(bool_t)); idx += sizeof(bool_t);
    std::memcpy(buffer.data()+idx, &n, sizeof(outer_t)); idx += sizeof(outer_t);
    std::memcpy(buffer.data()+idx, &s, sizeof(outer_t)); idx += sizeof(outer_t);
    const auto B_byte = static_cast<chunk_inner_t>(B);
    std::memcpy(buffer.data()+idx, &B_byte, sizeof(chunk_inner_t)); idx += sizeof(chunk_inner_t);
    std::memcpy(buffer.data()+idx, nnz.data(), sizeof(outer_t) * nnz.size()); idx += sizeof(outer_t) * nnz.size();

    char* const outer_ptr = buffer.data() + idx; idx += sizeof(outer_t) * (s+1); std::memcpy(outer_ptr, &idx, sizeof(outer_t));

    const auto outer_routine = [&](outer_t j) {
        outer_t snp_bytes = (2*B) * sizeof(outer_t);
        const auto cal0 = calldata.col(2*j);
        const auto cal1 = calldata.col(2*j+1);
        const auto anc0 = ancestries.col(2*j);
        const auto anc1 = ancestries.col(2*j+1);
        for (size_t bi = 0; bi < B; ++bi) {
            const auto a = sel[bi];
            snp_bytes += sizeof(inner_t);
            for (inner_t k = 0; k < max_chunks; ++k) {
                const outer_t base = k * chunk_size; inner_t nnz_chunk = 0; 
                for (inner_t c = 0; c < chunk_size; ++c) {
                    const outer_t didx = base + c; if (didx >= n) break;
                    bool h0 = (cal0[didx]==1 && anc0[didx]==static_cast<char>(a));
                    bool h1 = (cal1[didx]==1 && anc1[didx]==static_cast<char>(a));
                    if (h0) ++nnz_chunk;
                    if (h1) ++nnz_chunk;
                }
                if (nnz_chunk) snp_bytes += sizeof(inner_t) + sizeof(chunk_inner_t) + nnz_chunk * sizeof(chunk_inner_t);
            }
        }
        for (size_t bi = 0; bi < B; ++bi) {
            const auto a = sel[bi];
            snp_bytes += sizeof(inner_t);
            for (inner_t k = 0; k < max_chunks; ++k) {
                const outer_t base = k * chunk_size; inner_t nnz_chunk = 0; 
                for (inner_t c = 0; c < chunk_size; ++c) { const outer_t didx = base + c; if (didx >= n) break; if (anc0[didx]==static_cast<char>(a)) ++nnz_chunk; if (anc1[didx]==static_cast<char>(a)) ++nnz_chunk; }
                if (nnz_chunk) snp_bytes += sizeof(inner_t) + sizeof(chunk_inner_t) + nnz_chunk * sizeof(chunk_inner_t);
            }
        }
        std::memcpy(outer_ptr + sizeof(outer_t)*(j+1), &snp_bytes, sizeof(outer_t));
    };
    util::omp_parallel_for(outer_routine, 0, s, n_threads);

    for (outer_t j = 0; j < s; ++j) {
        const outer_t outer_curr = internal::read_as<outer_t>(outer_ptr + sizeof(outer_t)*(j+1));
        const outer_t outer_prev = internal::read_as<outer_t>(outer_ptr + sizeof(outer_t)*j);
        const outer_t sum = outer_curr + outer_prev; std::memcpy(outer_ptr + sizeof(outer_t)*(j+1), &sum, sizeof(outer_t));
    }
    const outer_t outer_last = internal::read_as<outer_t>(outer_ptr + sizeof(outer_t)*s);
    outer_t idx2 = outer_last;

    const auto inner_routine = [&](outer_t j) {
        const outer_t outer_curr = internal::read_as<outer_t>(outer_ptr + sizeof(outer_t)*(j+1));
        const outer_t outer_prev = internal::read_as<outer_t>(outer_ptr + sizeof(outer_t)*j);
        Eigen::Map<buffer_t> buffer_j(buffer.data() + outer_prev, outer_curr - outer_prev);
        outer_t cidx = (2*B) * sizeof(outer_t);
        const auto cal0 = calldata.col(2*j); const auto cal1 = calldata.col(2*j+1); const auto anc0 = ancestries.col(2*j); const auto anc1 = ancestries.col(2*j+1);
        for (size_t bi = 0; bi < B; ++bi) {
            const auto a = sel[bi];
            std::memcpy(buffer_j.data() + sizeof(outer_t)*bi, &cidx, sizeof(outer_t));
            auto *n_chunks_ptr = reinterpret_cast<inner_t*>(buffer_j.data() + cidx); cidx += sizeof(inner_t); inner_t n_chunks = 0;
            for (inner_t k = 0; k < max_chunks; ++k) {
                const outer_t base = k * chunk_size; size_t curr_idx = cidx;
                auto *chunk_index = reinterpret_cast<inner_t*>(buffer_j.data()+curr_idx); curr_idx += sizeof(inner_t);
                auto *chunk_nnz = reinterpret_cast<chunk_inner_t*>(buffer_j.data()+curr_idx); curr_idx += sizeof(chunk_inner_t);
                auto *chunk_begin = reinterpret_cast<chunk_inner_t*>(buffer_j.data()+curr_idx);
                inner_t nnz_chunk = 0;
                for (inner_t c = 0; c < chunk_size; ++c) {
                    const outer_t didx = base + c; if (didx >= n) break;
                    if (cal0[didx]==1 && anc0[didx]==static_cast<char>(a)) { chunk_begin[nnz_chunk++] = c; curr_idx += sizeof(chunk_inner_t); }
                    if (cal1[didx]==1 && anc1[didx]==static_cast<char>(a)) { chunk_begin[nnz_chunk++] = c; curr_idx += sizeof(chunk_inner_t); }
                }
                if (nnz_chunk) { std::memcpy(chunk_index, &k, sizeof(inner_t)); *chunk_nnz = nnz_chunk - 1; cidx = curr_idx; ++n_chunks; }
            }
            std::memcpy(n_chunks_ptr, &n_chunks, sizeof(inner_t));
        }
        for (size_t bi = 0; bi < B; ++bi) {
            const auto a = sel[bi];
            std::memcpy(buffer_j.data() + sizeof(outer_t)*(B + bi), &cidx, sizeof(outer_t));
            auto *n_chunks_ptr = reinterpret_cast<inner_t*>(buffer_j.data() + cidx); cidx += sizeof(inner_t); inner_t n_chunks = 0;
            for (inner_t k = 0; k < max_chunks; ++k) { const outer_t base = k * chunk_size; size_t curr_idx = cidx; auto *chunk_index = reinterpret_cast<inner_t*>(buffer_j.data()+curr_idx); curr_idx += sizeof(inner_t); auto *chunk_nnz = reinterpret_cast<chunk_inner_t*>(buffer_j.data()+curr_idx); curr_idx += sizeof(chunk_inner_t); auto *chunk_begin = reinterpret_cast<chunk_inner_t*>(buffer_j.data()+curr_idx); inner_t nnz_chunk = 0; for (inner_t c = 0; c < chunk_size; ++c) { const outer_t didx = base + c; if (didx >= n) break; if (anc0[didx]==static_cast<char>(a)) { chunk_begin[nnz_chunk++] = c; curr_idx += sizeof(chunk_inner_t); } if (anc1[didx]==static_cast<char>(a)) { chunk_begin[nnz_chunk++] = c; curr_idx += sizeof(chunk_inner_t); } } if (nnz_chunk) { std::memcpy(chunk_index, &k, sizeof(inner_t)); *chunk_nnz = nnz_chunk - 1; cidx = curr_idx; ++n_chunks; } }
            std::memcpy(n_chunks_ptr, &n_chunks, sizeof(inner_t));
        }
        if (cidx != static_cast<outer_t>(buffer_j.size())) { /* no-op safety */ }
    };
    util::omp_parallel_for(inner_routine, 0, s, n_threads);

    auto file_ptr = fopen_safe(_filename.c_str(), "wb");
    auto fp = file_ptr.get();
    auto total_bytes = std::fwrite(buffer.data(), sizeof(char), idx2, fp);
    if (total_bytes != static_cast<size_t>(idx2)) {
        throw util::adelie_core_error("Could not write the full buffer.");
    }
    return {total_bytes, {}};
}

} // namespace io
} // namespace adelie_core


