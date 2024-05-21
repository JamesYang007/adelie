#pragma once
#include <adelie_core/io/io_snp_base.hpp>
#include <adelie_core/io/io_snp_unphased.hpp>

namespace adelie_core {
namespace io {

template <class MmapPtrType=std::unique_ptr<char, std::function<void(char*)>>>
class IOSNPPhasedAncestry : public IOSNPBase<MmapPtrType>
{
public:
    using base_t = IOSNPBase<MmapPtrType>;
    using outer_t = uint64_t;
    using inner_t = uint32_t;
    using chunk_inner_t = u_char;
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
        1UL << (n_bits_per_byte * sizeof(chunk_inner_t))
    );
    static constexpr size_t n_haps = 2;

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

    outer_t ancestries() const 
    {
        if (!_is_read) throw_no_read();
        constexpr size_t idx = sizeof(bool_t) + 2 * sizeof(outer_t);
        return reinterpret_cast<const chunk_inner_t&>(_buffer[idx]);
    }

    outer_t cols() const
    {
        return snps() * ancestries();
    }

    Eigen::Ref<const vec_outer_t> nnz0() const 
    {
        if (!_is_read) throw_no_read();
        constexpr size_t slice = sizeof(bool_t) + 2 * sizeof(outer_t) + sizeof(chunk_inner_t);
        return Eigen::Map<const vec_outer_t>(
            reinterpret_cast<const outer_t*>(&_buffer[slice]),
            cols()
        );
    }

    Eigen::Ref<const vec_outer_t> nnz1() const 
    {
        if (!_is_read) throw_no_read();
        const size_t slice = sizeof(bool_t) + (2 + cols()) * sizeof(outer_t) + sizeof(chunk_inner_t);
        return Eigen::Map<const vec_outer_t>(
            reinterpret_cast<const outer_t*>(&_buffer[slice]),
            cols()
        );
    }

    Eigen::Ref<const vec_outer_t> outer() const
    {
        if (!_is_read) throw_no_read();
        const size_t slice = sizeof(bool_t) + 2 * (1 + cols()) * sizeof(outer_t) + sizeof(chunk_inner_t);
        return Eigen::Map<const vec_outer_t>(
            reinterpret_cast<const outer_t*>(&_buffer[slice]),
            snps() + 1
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

    const char* col_anc_hap(int j, int anc, int hap) const
    {
        const auto _col = col(j);
        const auto* _col_anc = (
            _col.data() + 
            reinterpret_cast<const outer_t*>(_col.data())[anc]
        );
        return (
            _col_anc + 
            reinterpret_cast<const outer_t*>(_col_anc)[hap]
        );
    }

    inner_t n_chunks(int j, int anc, int hap) const
    {
        const auto* _col_anc_hap = col_anc_hap(j, anc, hap);
        return *reinterpret_cast<const inner_t*>(_col_anc_hap);
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
    ) const
    {
        const size_t n = rows();
        const size_t s = snps();
        const size_t A = ancestries();
        rowarr_value_t dense(n, s * A);

        const auto routine = [&](outer_t j) {
            auto dense_j = dense.middleCols(A * j, A);
            dense_j.setZero();
            for (size_t a = 0; a < A; ++a) {
                for (size_t hap = 0; hap < n_haps; ++hap) {
                    auto it = this->begin(j, a, hap);
                    const auto end = this->end(j, a, hap);
                    for (; it != end; ++it) {
                        dense_j(*it, a) += 1;
                    }
                }
            }
        };
        if (n_threads <= 1) {
            for (outer_t j = 0; j < s; ++j) routine(j);
        } else {
            #pragma omp parallel for schedule(auto) num_threads(n_threads)
            for (outer_t j = 0; j < s; ++j) routine(j);
        }

        return dense;
    }

    std::tuple<size_t, std::unordered_map<std::string, double>> write(
        const Eigen::Ref<const colarr_value_t>& calldata,
        const Eigen::Ref<const colarr_value_t>& ancestries,
        size_t A,
        size_t n_threads
    ) const
    {
        using sw_t = util::Stopwatch;

        if (
            (calldata.rows() != ancestries.rows()) ||
            (calldata.cols() != ancestries.cols()) ||
            (calldata.cols() % n_haps)
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
        const outer_t s = calldata.cols() / n_haps;

        const size_t max_chunks = (n + chunk_size - 1) / chunk_size;
        if (max_chunks >= _max_inner) {
            throw util::adelie_core_error(
                "calldata dimensions are too large! "
            );
        } 

        // compute number of non-zero values
        vec_outer_t nnz0(s * A);
        vec_outer_t nnz1(s * A);
        sw.start();
        compute_nnz(
            colarr_value_t::NullaryExpr(n, s * A, [&](auto i, auto j) {
                const auto snp = j / A;
                const auto anc = j % A;
                const auto k = 2 * snp;
                return calldata(i, k) && (ancestries(i, k) == anc);
            }),
            nnz0, 
            n_threads
        );
        compute_nnz(
            colarr_value_t::NullaryExpr(n, s * A, [&](auto i, auto j) {
                const auto snp = j / A;
                const auto anc = j % A;
                const auto k = 2 * snp + 1;
                return calldata(i, k) && (ancestries(i, k) == anc);
            }),
            nnz1, 
            n_threads
        );
        benchmark["nnz"] = sw.elapsed();

        // allocate sufficient memory (upper bound on size)
        const size_t preamble_size = (
            sizeof(bool_t) +                    // endian
            2 * sizeof(outer_t) +               // n, s
            sizeof(chunk_inner_t) +             // A
            (nnz0.size() + nnz1.size()) * sizeof(outer_t) + // nnz0, nnz1
            (s + 1) * sizeof(outer_t)           // outer (snps)
        );
        buffer_t buffer(
            preamble_size +
            s * A * (               // for each snp, ancestry
                sizeof(outer_t) +       // outer (ancestry)
                n_haps * (              // for each hap
                    sizeof(outer_t) +       // outer (hap)
                    sizeof(inner_t) +       // n_chunks
                    max_chunks * (          // for each chunk
                        sizeof(inner_t) +       // chunk index
                        sizeof(chunk_inner_t)   // chunk nnz - 1
                    )
                )
            ) +
            (nnz0.sum() + nnz1.sum()) * sizeof(chunk_inner_t)   // nnz * char
        );

        // populate buffer
        size_t idx = 0;
        reinterpret_cast<bool_t&>(buffer[idx]) = endian; idx += sizeof(bool_t);
        reinterpret_cast<outer_t&>(buffer[idx]) = n; idx += sizeof(outer_t);
        reinterpret_cast<outer_t&>(buffer[idx]) = s; idx += sizeof(outer_t);
        reinterpret_cast<chunk_inner_t&>(buffer[idx]) = A; idx += sizeof(chunk_inner_t);
        Eigen::Map<vec_outer_t>(
            reinterpret_cast<outer_t*>(&buffer[idx]),
            nnz0.size()
        ) = nnz0; idx += sizeof(outer_t) * nnz0.size();
        Eigen::Map<vec_outer_t>(
            reinterpret_cast<outer_t*>(&buffer[idx]),
            nnz1.size()
        ) = nnz1; idx += sizeof(outer_t) * nnz1.size();

        // outer[i] = number of bytes to jump from beginning of file 
        // to start reading snp i.
        // outer[i+1] - outer[i] = total number of bytes for snp i. 
        Eigen::Map<vec_outer_t> outer(
            reinterpret_cast<outer_t*>(&buffer[idx]),
            s + 1
        ); idx += sizeof(outer_t) * outer.size();
        outer[0] = idx;

        // populate outer 
        const auto outer_routine = [&](outer_t j) {
            size_t snp_bytes = 0;
            for (size_t a = 0; a < A; ++a) {
                snp_bytes += sizeof(outer_t);
                for (size_t hap = 0; hap < n_haps; ++hap) {
                    snp_bytes += sizeof(outer_t) + sizeof(inner_t);
                    const auto k = n_haps * j + hap;
                    const auto cal_jh = calldata.col(k);
                    const auto anc_jh = ancestries.col(k);

                    for (inner_t k = 0; k < max_chunks; ++k) {
                        const outer_t chnk = k * chunk_size;
                        bool is_nonempty = false;
                        for (inner_t c = 0; c < chunk_size; ++c) {
                            const outer_t cidx = chnk + c;
                            if (cidx >= n) break;
                            if ((anc_jh[cidx] < 0) || (anc_jh[cidx] >= A)) {
                                throw util::adelie_core_error(
                                    "Detected an ancestry not in the range [0,A):"
                                    "\n\tancestries[" + std::to_string(cidx) +
                                    ", " + std::to_string(k) +
                                    "] = " + std::to_string(anc_jh[cidx]) +
                                    "\nMake sure ancestries only contains values in [0,A)."
                                );
                            }
                            if ((cal_jh[cidx] != 0) && (cal_jh[cidx] != 1)) {
                                throw util::adelie_core_error(
                                    "Detected a non-binary value: "
                                    "\n\tcalldata[" + std::to_string(cidx) +
                                    ", " + std::to_string(k) +
                                    "] = " + std::to_string(cal_jh[cidx]) +
                                    "\nMake sure calldata only contains 0 or 1 values."
                                );
                            }
                            // always error check (above) before proceeding
                            const bool to_not_skip = (
                                (anc_jh[cidx] == a) && 
                                (cal_jh[cidx] == 1)
                            );
                            if (!to_not_skip) continue;
                            is_nonempty = true;
                            snp_bytes += sizeof(char);
                        }
                        snp_bytes += is_nonempty * (sizeof(inner_t) + sizeof(chunk_inner_t));
                    }
                }
            }
            outer[j+1] = snp_bytes;
        };
        sw.start();
        if (n_threads <= 1) {
            for (outer_t j = 0; j < s; ++j) outer_routine(j);
        } else {
            #pragma omp parallel for schedule(auto) num_threads(n_threads)
            for (outer_t j = 0; j < s; ++j) outer_routine(j);
        }
        benchmark["outer_time"] = sw.elapsed();

        // cumsum outer
        for (outer_t j = 0; j < s; ++j) outer[j+1] += outer[j];

        if (outer[s] > buffer.size()) {
            throw util::adelie_core_error(
                "Buffer was not initialized with a large enough size. "
                "\n\tBuffer size:   " + std::to_string(buffer.size()) +
                "\n\tExpected size: " + std::to_string(outer[s]) +
                "\nThis is likely a bug in the code. Please report it! "
            );
        }
        idx = outer[s];

        // populate (column) inner buffers
        const auto inner_routine = [&](outer_t j) {
            Eigen::Map<buffer_t> buffer_j(
                reinterpret_cast<char*>(&buffer[outer[j]]),
                outer[j+1] - outer[j]
            );
            size_t cidx = A * sizeof(outer_t);

            for (size_t a = 0; a < A; ++a) {
                auto& outer_anc = reinterpret_cast<outer_t*>(buffer_j.data())[a];
                outer_anc = cidx;

                Eigen::Map<buffer_t> buffer_jh(
                    buffer_j.data() + cidx,
                    buffer_j.size() - cidx
                );
                size_t hcidx = n_haps * sizeof(outer_t);

                for (size_t hap = 0; hap < n_haps; ++hap) {
                    const auto k = n_haps * j + hap;
                    const auto cal_jh = calldata.col(k);
                    const auto anc_jh = ancestries.col(k);
                    auto& outer_hap = reinterpret_cast<outer_t*>(buffer_jh.data())[hap];
                    outer_hap = hcidx;

                    auto& n_chunks = reinterpret_cast<inner_t&>(buffer_jh[hcidx]);
                    hcidx += sizeof(inner_t);
                    n_chunks = 0;

                    for (inner_t k = 0; k < max_chunks; ++k) {
                        const outer_t chnk = k * chunk_size;
                        size_t curr_idx = hcidx;
                        auto* chunk_index = reinterpret_cast<inner_t*>(buffer_jh.data() + curr_idx); 
                        curr_idx += sizeof(inner_t);
                        auto* chunk_nnz = reinterpret_cast<chunk_inner_t*>(buffer_jh.data() + curr_idx); 
                        curr_idx += sizeof(chunk_inner_t);
                        auto* chunk_begin = reinterpret_cast<chunk_inner_t*>(buffer_jh.data() + curr_idx); 
                        inner_t nnz = 0;
                        for (inner_t c = 0; c < chunk_size; ++c) {
                            const outer_t didx = chnk + c;
                            if (didx >= n) break;
                            const bool to_not_skip = (
                                (anc_jh[didx] == a) && 
                                (cal_jh[didx] == 1)
                            );
                            if (!to_not_skip) continue;
                            chunk_begin[nnz] = c;
                            ++nnz;
                            curr_idx += sizeof(chunk_inner_t);
                        }
                        if (nnz) {
                            *chunk_index = k;
                            *chunk_nnz = nnz - 1;
                            hcidx = curr_idx;
                            ++n_chunks;
                        }
                    }
                }
                cidx += hcidx;
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
            for (outer_t j = 0; j < s; ++j) inner_routine(j);
        } else {
            #pragma omp parallel for schedule(auto) num_threads(n_threads)
            for (outer_t j = 0; j < s; ++j) inner_routine(j);
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