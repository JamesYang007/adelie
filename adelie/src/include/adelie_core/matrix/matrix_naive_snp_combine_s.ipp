#pragma once
#include <string>
#include <vector>
#include <adelie_core/matrix/matrix_naive_snp_combine_s.hpp>
#include <adelie_core/configs.hpp>
#include <adelie_core/util/omp.hpp>

namespace adelie_core {
namespace matrix {

namespace {

template <class IO, class V, class Buff>
auto both_dot(const IO& io, int j, const V& v, size_t n_threads, Buff& buff)
{
    using value_t = typename std::decay_t<V>::Scalar;
    const auto A = io.ancestries();
    const auto snp = j / (2 * A);
    const auto col_in_block = j % (2 * A);
    const auto nnz = io.nnz()[j];
    const size_t n_bytes = (8 * sizeof(value_t)) * nnz;
    if (n_threads <= 1 || util::omp_in_parallel() || n_bytes <= Configs::min_bytes) {
        value_t sum = 0;
        auto it = io.begin(snp, col_in_block);
        const auto end = io.end(snp, col_in_block);
        for (; it != end; ++it) sum += v[*it];
        return sum;
    }
    auto vbuff = buff.head(n_threads);
    vbuff.setZero();
    const size_t n_chunks = io.n_chunks(snp, col_in_block);
    const int n_blocks = std::min<int>(n_threads, static_cast<int>(n_chunks));
    if (n_blocks > 0) {
        const int block_size = static_cast<int>(n_chunks) / n_blocks;
        const int remainder = static_cast<int>(n_chunks) % n_blocks;
        #pragma omp parallel for schedule(static) num_threads(n_threads)
        for (int t = 0; t < n_blocks; ++t) {
            const auto begin = (
                std::min<int>(t, remainder) * (block_size + 1)
                + std::max<int>(t - remainder, 0) * block_size
            );
            const auto size = block_size + (t < remainder);
            auto it = io.begin(snp, col_in_block, begin);
            const auto end = io.begin(snp, col_in_block, begin + size);
            value_t sum = 0;
            for (; it != end; ++it) sum += v[*it];
            vbuff[t] = sum;
        }
    }
    return vbuff.sum();
}

template <class IO, class VT, class Out>
void both_axi(const IO& io, int j, VT v, Out& out, size_t n_threads)
{
    using value_t = VT;
    const auto A = io.ancestries();
    const auto snp = j / (2 * A);
    const auto col_in_block = j % (2 * A);
    const auto nnz = io.nnz()[j];
    const size_t n_bytes = (4 * sizeof(value_t)) * nnz;
    if (n_threads <= 1 || util::omp_in_parallel() || n_bytes <= Configs::min_bytes) {
        auto it = io.begin(snp, col_in_block);
        const auto end = io.end(snp, col_in_block);
        for (; it != end; ++it) out[*it] += v;
        return;
    }
    const size_t n_chunks = io.n_chunks(snp, col_in_block);
    const int n_blocks = std::min<int>(n_threads, static_cast<int>(n_chunks));
    if (n_blocks > 0) {
        const int block_size = static_cast<int>(n_chunks) / n_blocks;
        const int remainder = static_cast<int>(n_chunks) % n_blocks;
        #pragma omp parallel for schedule(static) num_threads(n_threads)
        for (int t = 0; t < n_blocks; ++t) {
            const auto begin = (
                std::min<int>(t, remainder) * (block_size + 1)
                + std::max<int>(t - remainder, 0) * block_size
            );
            const auto size = block_size + (t < remainder);
            auto it = io.begin(snp, col_in_block, begin);
            const auto end = io.begin(snp, col_in_block, begin + size);
            for (; it != end; ++it) out[*it] += v;
        }
    }
}

} // anonymous namespace

ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_S_TP
typename ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_S::value_t
ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_S::_cmul(
    int j,
    const Eigen::Ref<const vec_value_t>& v,
    const Eigen::Ref<const vec_value_t>& weights,
    size_t n_threads,
    Eigen::Ref<vec_value_t> buff
) const 
{
    return both_dot(_io, j, v * weights, n_threads, buff);
}

ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_S_TP
typename ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_S::value_t
ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_S::_sq_cmul(
    int j,
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> /*buff*/
) const 
{
    using value_t = typename std::decay_t<decltype(weights)>::Scalar;

    const auto A = _io.ancestries();
    const auto snp = j / (2 * A);
    const auto col_in_block = j % (2 * A);

    // Count dosages for each sample
    std::vector<int> dosages(_io.rows(), 0);

    // Both types of columns are simple presence counts
    auto it = _io.begin(snp, col_in_block);
    const auto end = _io.end(snp, col_in_block);
    for (; it != end; ++it) {
        dosages[*it]++;
    }

    value_t sum = 0;
    for (size_t i = 0; i < dosages.size(); ++i) {
        const auto dosage = dosages[i];
        sum += weights[i] * dosage * dosage;
    }
    return sum;
}

ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_S_TP
void
ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_S::_ctmul(
    int j,
    value_t v,
    Eigen::Ref<vec_value_t> out,
    size_t n_threads
) const
{
    return both_axi(_io, j, v, out, n_threads);
}

ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_S_TP
ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_S::MatrixNaiveSNPCombineS(
    const io_t& io,
    size_t n_threads
): 
    _io(io),
    _n_threads(n_threads),
    _buff(n_threads * (2 * _io.ancestries()))
{
    if (n_threads < 1) {
        throw util::adelie_core_error("n_threads must be >= 1.");
    }
    if (_io.ancestries() < 1) {
        throw util::adelie_core_error("Number of ancestries must be >= 1.");
    }
}

ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_S_TP
typename ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_S::value_t
ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_S::cmul(
    int j, 
    const Eigen::Ref<const vec_value_t>& v,
    const Eigen::Ref<const vec_value_t>& weights
)
{
    base_t::check_cmul(j, v.size(), weights.size(), rows(), cols());
    return _cmul(j, v, weights, _n_threads, _buff);
}

ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_S_TP
typename ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_S::value_t
ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_S::cmul_safe(
    int j, 
    const Eigen::Ref<const vec_value_t>& v,
    const Eigen::Ref<const vec_value_t>& weights
) const
{
    base_t::check_cmul(j, v.size(), weights.size(), rows(), cols());
    vec_value_t buff(_n_threads * (_n_threads > 1) * !util::omp_in_parallel());
    return _cmul(j, v, weights, _n_threads, buff);
}

ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_S_TP
void
ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_S::ctmul(
    int j, 
    value_t v, 
    Eigen::Ref<vec_value_t> out
)
{
    base_t::check_ctmul(j, out.size(), rows(), cols());
    _ctmul(j, v, out, _n_threads);
}

ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_S_TP
void
ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_S::bmul(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
)
{
    base_t::check_bmul(j, q, v.size(), weights.size(), out.size(), rows(), cols());
    if (static_cast<size_t>(_buff.size()) < q * _n_threads) _buff.resize(q * _n_threads);
    // Compute column-wise dot products independently
    for (int k = 0; k < q; ++k) {
        out[k] = both_dot(_io, j + k, v * weights, _n_threads, _buff);
    }
}

ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_S_TP
void
ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_S::bmul_safe(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
{
    base_t::check_bmul(j, q, v.size(), weights.size(), out.size(), rows(), cols());
    vec_value_t buff(q * _n_threads * (_n_threads > 1) * !util::omp_in_parallel());
    for (int k = 0; k < q; ++k) {
        out[k] = both_dot(_io, j + k, v * weights, _n_threads, buff);
    }
}

ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_S_TP
void
ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_S::btmul(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    Eigen::Ref<vec_value_t> out
)
{
    base_t::check_btmul(j, q, v.size(), out.size(), rows(), cols());
    for (int k = 0; k < q; ++k) {
        both_axi(_io, j + k, v[k], out, _n_threads);
    }
}

ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_S_TP
void
ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_S::mul(
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
{
    const auto routine = [&](int t) {
        out[t] = _cmul(t, v, weights, 1, out /* unused */);
    };
    util::omp_parallel_for(routine, 0, cols(), _n_threads);
}

ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_S_TP
void
ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_S::cov(
    int j, int q,
    const Eigen::Ref<const vec_value_t>& sqrt_weights,
    Eigen::Ref<colmat_value_t> out
) const
{
    base_t::check_cov(
        j, q, sqrt_weights.size(), 
        out.rows(), out.cols(),
        rows(), cols()
    );
    out.setZero();

    const auto A = ancestries();

    // Temporary buffers, reused across columns
    util::rowvec_type<char> bbuff(_io.rows());
    vec_index_t ibuff(_io.rows());
    vec_value_t buff(_n_threads * (_n_threads > 1) * !util::omp_in_parallel());
    bbuff.setZero();

    int n_solved0 = 0;
    while (n_solved0 < q) {
        const auto begin0 = j + n_solved0;
        const auto snp0 = begin0 / (2 * A);
        const auto col_lower0 = begin0 % (2 * A);
        const auto col_upper0 = std::min<int>(col_lower0 + q - n_solved0, 2 * A);

        int n_solved1 = 0;
        while (n_solved1 <= n_solved0) {
            const auto begin1 = j + n_solved1;
            const auto col_lower1 = begin1 % (2 * A);
            const auto col_upper1 = std::min<int>(col_lower1 + q - n_solved1, 2 * A);

            const auto col_size0 = col_upper0 - col_lower0;
            const auto col_size1 = col_upper1 - col_lower1;

            for (size_t c0 = 0; c0 < static_cast<size_t>(col_size0); ++c0) {
                const auto col_in_block0 = col_lower0 + static_cast<int>(c0);
                size_t nnz = 0;

                // Build per-row multiplicity for column 0 within this block
                auto it = _io.begin(snp0, col_in_block0);
                const auto end = _io.end(snp0, col_in_block0);
                for (; it != end; ++it) {
                    const auto idx = *it;
                    if (!bbuff[idx]) {
                        ibuff[nnz] = static_cast<int>(idx);
                        ++nnz;
                    }
                    bbuff[idx] += 1;
                }

                // Cross terms with columns in the second block window
                for (size_t c1 = 0; c1 < static_cast<size_t>(col_size1); ++c1) {
                    // weighted dot using v[i] = w[i] * bbuff[i]
                    const auto sum = both_dot(
                        _io,
                        begin1 + static_cast<int>(c1),
                        vec_value_t::NullaryExpr(sqrt_weights.size(), [&](auto i) {
                            const auto sw = sqrt_weights[i];
                            return sw * sw * static_cast<typename vec_value_t::Scalar>(bbuff[i]);
                        }),
                        _n_threads,
                        buff
                    );
                    const auto kk0 = n_solved0 + static_cast<int>(c0);
                    const auto kk1 = n_solved1 + static_cast<int>(c1);
                    if (kk0 == kk1) {
                        out(kk0, kk0) = sum;
                    } else {
                        out(kk0, kk1) = sum;
                        out(kk1, kk0) = sum;
                    }
                }

                // reset bbuff entries we touched
                for (size_t i = 0; i < nnz; ++i) {
                    bbuff[ibuff[i]] = 0;
                }
            }

            n_solved1 += col_upper1 - col_lower1;
        }
        n_solved0 += col_upper0 - col_lower0;
    }
}

ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_S_TP
void
ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_S::mean(
    const Eigen::Ref<const vec_value_t>&,
    Eigen::Ref<vec_value_t> out
) const
{
    out.setZero();
}

ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_S_TP
void
ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_S::var(
    const Eigen::Ref<const vec_value_t>&,
    const Eigen::Ref<const vec_value_t>&,
    Eigen::Ref<vec_value_t> out
) const
{
    out.setOnes();
}

ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_S_TP
int
ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_S::rows() const 
{ 
    return _io.rows(); 
}

ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_S_TP
int
ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_S::cols() const 
{ 
    return _io.snps() * (2 * ancestries()); 
}

ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_S_TP
void
ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_S::sq_mul(
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
{
    const auto routine = [&](int t) {
        out[t] = _sq_cmul(t, weights, out /* unused */);
    };
    util::omp_parallel_for(routine, 0, cols(), _n_threads);
}

ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_S_TP
void
ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_S::sp_tmul(
    const sp_mat_value_t& v,
    Eigen::Ref<rowmat_value_t> out
) const
{
    base_t::check_sp_tmul(
        v.rows(), v.cols(), out.rows(), out.cols(), rows(), cols()
    );
    const auto routine = [&](int k) {
        typename sp_mat_value_t::InnerIterator it(v, k);
        auto out_k = out.row(k);
        out_k.setZero();
        for (; it; ++it) {
            _ctmul(it.index(), it.value(), out_k, 1);
        }
    };
    util::omp_parallel_for(routine, 0, v.outerSize(), _n_threads);
}

} // namespace matrix
} // namespace adelie_core


