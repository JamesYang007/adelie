#pragma once
#include <string>
#include <vector>
#include <adelie_core/matrix/matrix_naive_snp_combine_r.hpp>
#include <adelie_core/matrix/utils.hpp>

namespace adelie_core {
namespace matrix {

ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_R_TP
typename ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_R::value_t
ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_R::_cmul(
    int j,
    const Eigen::Ref<const vec_value_t>& v,
    const Eigen::Ref<const vec_value_t>& weights,
    size_t n_threads,
    Eigen::Ref<vec_value_t> buff
) const 
{
    return snp_combine_r_dot(
        _io, j, v * weights, n_threads, buff
    );
}

ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_R_TP
typename ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_R::value_t
ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_R::_sq_cmul(
    int j,
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> /*buff*/
) const 
{
    using value_t = typename std::decay_t<decltype(weights)>::Scalar;

    const auto A = _io.ancestries();
    const auto snp = j / (1 + A);
    const auto col_in_block = j % (1 + A);
    
    // Count dosages for each sample
    std::vector<int> dosages(_io.rows(), 0);
    
    if (col_in_block == 0) {
        // SNP data column
        auto it = _io.begin(snp, 0);
        const auto end = _io.end(snp, 0);
        for (; it != end; ++it) {
            dosages[*it]++;
        }
    } else {
        // Ancestry column
        const auto anc = col_in_block;
        auto it = _io.begin(snp, anc);
        const auto end = _io.end(snp, anc);
        for (; it != end; ++it) {
            dosages[*it]++;
        }
    }
    
    // Compute weighted sum of squared dosages
    value_t sum = 0;
    for (size_t i = 0; i < dosages.size(); ++i) {
        const auto dosage = dosages[i];
        sum += weights[i] * dosage * dosage;
    }
    
    return sum;
}

ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_R_TP
void
ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_R::_ctmul(
    int j,
    value_t v,
    Eigen::Ref<vec_value_t> out,
    size_t n_threads
) const
{
    return snp_combine_r_axi(
        _io, j, v, out, n_threads
    );
}

ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_R_TP
ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_R::MatrixNaiveSNPCombineR(
    const io_t& io,
    size_t n_threads
): 
    _io(io),
    _n_threads(n_threads),
    _buff(n_threads * (1 + _io.ancestries()))
{
    if (n_threads < 1) {
        throw util::adelie_core_error("n_threads must be >= 1.");
    }
    if (_io.ancestries() < 1) {
        throw util::adelie_core_error("Number of ancestries must be >= 1.");
    }
}

ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_R_TP
typename ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_R::value_t
ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_R::cmul(
    int j, 
    const Eigen::Ref<const vec_value_t>& v,
    const Eigen::Ref<const vec_value_t>& weights
)
{
    base_t::check_cmul(j, v.size(), weights.size(), rows(), cols());
    return _cmul(j, v, weights, _n_threads, _buff);
}

ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_R_TP
typename ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_R::value_t
ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_R::cmul_safe(
    int j, 
    const Eigen::Ref<const vec_value_t>& v,
    const Eigen::Ref<const vec_value_t>& weights
) const
{
    base_t::check_cmul(j, v.size(), weights.size(), rows(), cols());
    vec_value_t buff(_n_threads * (_n_threads > 1) * !util::omp_in_parallel());
    return _cmul(j, v, weights, _n_threads, buff);
}

ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_R_TP
void
ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_R::ctmul(
    int j, 
    value_t v, 
    Eigen::Ref<vec_value_t> out
)
{
    base_t::check_ctmul(j, out.size(), rows(), cols());
    _ctmul(j, v, out, _n_threads);
}

ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_R_TP
void
ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_R::bmul(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
)
{
    base_t::check_bmul(j, q, v.size(), weights.size(), out.size(), rows(), cols());
    if (static_cast<size_t>(_buff.size()) < q * _n_threads) _buff.resize(q * _n_threads);
    snp_combine_r_block_dot(
        _io, j, q, v * weights, out, _n_threads, _buff
    );
}

ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_R_TP
void
ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_R::bmul_safe(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
{
    base_t::check_bmul(j, q, v.size(), weights.size(), out.size(), rows(), cols());
    vec_value_t buff(q * _n_threads * (_n_threads > 1) * !util::omp_in_parallel());
    snp_combine_r_block_dot(
        _io, j, q, v * weights, out, _n_threads, buff
    );
}

ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_R_TP
void
ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_R::btmul(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    Eigen::Ref<vec_value_t> out
)
{
    base_t::check_btmul(j, q, v.size(), out.size(), rows(), cols());
    snp_combine_r_block_axi(
        _io, j, q, v, out, _n_threads
    );
}

ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_R_TP
void
ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_R::mul(
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

ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_R_TP
void
ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_R::cov(
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
    
    const auto A = ancestries();

    out.setZero(); // don't parallelize! q is usually small

    util::rowvec_type<char> bbuff(_io.rows());
    vec_index_t ibuff(_io.rows());
    vec_value_t buff(_n_threads * (_n_threads > 1) * !util::omp_in_parallel());
    bbuff.setZero();

    int n_solved0 = 0;
    while (n_solved0 < q) {
        const auto begin0 = j + n_solved0;
        const auto snp0 = begin0 / (1 + A);
        const auto col_lower0 = begin0 % (1 + A);
        const auto col_upper0 = std::min<int>(col_lower0 + q - n_solved0, 1 + A);

        int n_solved1 = 0;
        while (n_solved1 <= n_solved0) {
            const auto begin1 = j + n_solved1;
            const auto col_lower1 = begin1 % (1 + A);
            const auto col_upper1 = std::min<int>(col_lower1 + q - n_solved1, 1 + A);

            #if 0
            // This block of code will not be compiled.
            if (n_solved0 == n_solved1) {
                const auto snp = snp0;
                const auto c_low = col_lower0;
                const auto c_high = col_upper0;
                const auto c_size = c_high - c_low;

                // increase buffer including cross-term computation part as well
                if (
                    static_cast<size_t>(buff.size()) < c_size * _n_threads &&
                    _n_threads > 1 &&
                    !util::omp_in_parallel()
                ) {
                    buff.resize(c_size * _n_threads);
                }

                const auto routine = [&](auto k0, auto k1) {
                    const auto sum = snp_combine_r_cross_dot(
                        _io,
                        snp * (1 + A) + c_low + k0,
                        snp * (1 + A) + c_low + k1,
                        sqrt_weights.square()
                    );
                    out(n_solved0 + k0, n_solved0 + k1) = sum;
                };

                if (_n_threads <= 1 || util::omp_in_parallel()) {
                    for (int k0 = 0; k0 < static_cast<int>(c_size); ++k0) {
                        for (int k1 = 0; k1 < static_cast<int>(c_size); ++k1) {
                            routine(k0, k1);
                        }
                    }
                } else {
                    #pragma omp parallel for schedule(static) num_threads(_n_threads) collapse(2)
                    for (int k0 = 0; k0 < static_cast<int>(c_size); ++k0) {
                        for (int k1 = 0; k1 < static_cast<int>(c_size); ++k1) {
                            routine(k0, k1);
                        }
                    }
                }

                n_solved1 += col_upper1 - col_lower1;
                continue;
            }
            #endif

            /* general routine */

            const auto col_size0 = col_upper0 - col_lower0;
            const auto col_size1 = col_upper1 - col_lower1;
            for (size_t c0 = 0; c0 < col_size0; ++c0) {
                const auto col_in_block0 = col_lower0 + c0;
                size_t nnz = 0;
                
                if (col_in_block0 == 0) {
                    // SNP data column
                    auto it = _io.begin(snp0, 0);
                    const auto end = _io.end(snp0, 0);
                    for (; it != end; ++it) {
                        const auto idx = *it;
                        if (!bbuff[idx]) {
                            ibuff[nnz] = idx;
                            ++nnz;
                        }
                        bbuff[idx] += 1;
                    }
                } else {
                    // Ancestry column
                    auto it = _io.begin(snp0, col_in_block0);
                    const auto end = _io.end(snp0, col_in_block0);
                    for (; it != end; ++it) {
                        const auto idx = *it;
                        if (!bbuff[idx]) {
                            ibuff[nnz] = idx;
                            ++nnz;
                        }
                        bbuff[idx] += 1;
                    }
                }

                for (size_t c1 = 0; c1 < col_size1; ++c1) {
                    const auto sum = snp_combine_r_dot(
                        _io, begin1 + c1, 
                        vec_value_t::NullaryExpr(sqrt_weights.size(), [&](auto i) {
                            const auto sqrt_wi = sqrt_weights[i];
                            return sqrt_wi * sqrt_wi * bbuff[i];
                        }),
                        _n_threads,
                        buff
                    );
                    const auto kk0 = n_solved0 + c0;
                    const auto kk1 = n_solved1 + c1;
                    // --- assign symmetrically, no accumulation ---
                    if (kk0 == kk1) {
                        // diagonal
                        out(kk0, kk0) = sum;
                    } else {
                        // off–diagonal
                        out(kk0, kk1) = sum;
                        out(kk1, kk0) = sum;
                    }
                }

                for (size_t i = 0; i < nnz; ++i) {
                    bbuff[ibuff[i]] = 0;
                }
            }

            n_solved1 += col_upper1 - col_lower1;
        }
        n_solved0 += col_upper0 - col_lower0;
    }     
}

ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_R_TP
int
ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_R::rows() const 
{ 
    return _io.rows(); 
}

ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_R_TP
int
ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_R::cols() const 
{ 
    return _io.snps() * (1 + ancestries()); 
}

ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_R_TP
void
ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_R::sq_mul(
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
{
    const auto routine = [&](int t) {
        out[t] = _sq_cmul(t, weights, out /* unused */);
    };
    util::omp_parallel_for(routine, 0, cols(), _n_threads);
}

ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_R_TP
void
ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_R::sp_tmul(
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

ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_R_TP
void
ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_R::mean(
    const Eigen::Ref<const vec_value_t>&,
    Eigen::Ref<vec_value_t> out
) const
{
    out.setZero();
}

ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_R_TP
void
ADELIE_CORE_MATRIX_NAIVE_SNP_COMBINE_R::var(
    const Eigen::Ref<const vec_value_t>&,
    const Eigen::Ref<const vec_value_t>&,
    Eigen::Ref<vec_value_t> out
) const
{
    out.setOnes();
}

} // namespace matrix
} // namespace adelie_core