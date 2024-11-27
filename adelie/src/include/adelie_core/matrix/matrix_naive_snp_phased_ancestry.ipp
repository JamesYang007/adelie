#pragma once
#include <string>
#include <vector>
#include <adelie_core/matrix/matrix_naive_snp_phased_ancestry.hpp>
#include <adelie_core/matrix/utils.hpp>

namespace adelie_core {
namespace matrix {

ADELIE_CORE_MATRIX_NAIVE_SNP_PHASED_ANCESTRY_TP
typename ADELIE_CORE_MATRIX_NAIVE_SNP_PHASED_ANCESTRY::value_t
ADELIE_CORE_MATRIX_NAIVE_SNP_PHASED_ANCESTRY::_cmul(
    int j,
    const Eigen::Ref<const vec_value_t>& v,
    const Eigen::Ref<const vec_value_t>& weights,
    size_t n_threads,
    Eigen::Ref<vec_value_t> buff
) const 
{
    return snp_phased_ancestry_dot(
        _io, j, v * weights, n_threads, buff
    );
}

ADELIE_CORE_MATRIX_NAIVE_SNP_PHASED_ANCESTRY_TP
typename ADELIE_CORE_MATRIX_NAIVE_SNP_PHASED_ANCESTRY::value_t
ADELIE_CORE_MATRIX_NAIVE_SNP_PHASED_ANCESTRY::_sq_cmul(
    int j,
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> buff
) const 
{
    constexpr size_t n_threads = 1;
    const auto sum = snp_phased_ancestry_dot(_io, j, weights, n_threads, buff);
    const auto cross_sum = snp_phased_ancestry_cross_dot(_io, j, j, weights);
    return sum + 2 * cross_sum;
}

ADELIE_CORE_MATRIX_NAIVE_SNP_PHASED_ANCESTRY_TP
void
ADELIE_CORE_MATRIX_NAIVE_SNP_PHASED_ANCESTRY::_ctmul(
    int j,
    value_t v,
    Eigen::Ref<vec_value_t> out,
    size_t n_threads
) const
{
    return snp_phased_ancestry_axi(
        _io, j, v, out, n_threads
    );
}

ADELIE_CORE_MATRIX_NAIVE_SNP_PHASED_ANCESTRY_TP
ADELIE_CORE_MATRIX_NAIVE_SNP_PHASED_ANCESTRY::MatrixNaiveSNPPhasedAncestry(
    const io_t& io,
    size_t n_threads
): 
    _io(io),
    _n_threads(n_threads),
    _buff(n_threads * _io.ancestries())
{
    if (n_threads < 1) {
        throw util::adelie_core_error("n_threads must be >= 1.");
    }
    if (_io.ancestries() < 1) {
        throw util::adelie_core_error("Number of ancestries must be >= 1.");
    }
}

ADELIE_CORE_MATRIX_NAIVE_SNP_PHASED_ANCESTRY_TP
typename ADELIE_CORE_MATRIX_NAIVE_SNP_PHASED_ANCESTRY::value_t
ADELIE_CORE_MATRIX_NAIVE_SNP_PHASED_ANCESTRY::cmul(
    int j, 
    const Eigen::Ref<const vec_value_t>& v,
    const Eigen::Ref<const vec_value_t>& weights
)
{
    base_t::check_cmul(j, v.size(), weights.size(), rows(), cols());
    return _cmul(j, v, weights, _n_threads, _buff);
}

ADELIE_CORE_MATRIX_NAIVE_SNP_PHASED_ANCESTRY_TP
typename ADELIE_CORE_MATRIX_NAIVE_SNP_PHASED_ANCESTRY::value_t
ADELIE_CORE_MATRIX_NAIVE_SNP_PHASED_ANCESTRY::cmul_safe(
    int j, 
    const Eigen::Ref<const vec_value_t>& v,
    const Eigen::Ref<const vec_value_t>& weights
) const
{
    base_t::check_cmul(j, v.size(), weights.size(), rows(), cols());
    vec_value_t buff(_n_threads * (_n_threads > 1) * !util::omp_in_parallel());
    return _cmul(j, v, weights, _n_threads, buff);
}

ADELIE_CORE_MATRIX_NAIVE_SNP_PHASED_ANCESTRY_TP
void
ADELIE_CORE_MATRIX_NAIVE_SNP_PHASED_ANCESTRY::ctmul(
    int j, 
    value_t v, 
    Eigen::Ref<vec_value_t> out
)
{
    base_t::check_ctmul(j, out.size(), rows(), cols());
    _ctmul(j, v, out, _n_threads);
}

ADELIE_CORE_MATRIX_NAIVE_SNP_PHASED_ANCESTRY_TP
void
ADELIE_CORE_MATRIX_NAIVE_SNP_PHASED_ANCESTRY::bmul(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
)
{
    base_t::check_bmul(j, q, v.size(), weights.size(), out.size(), rows(), cols());
    if (static_cast<size_t>(_buff.size()) < q * _n_threads) _buff.resize(q * _n_threads);
    snp_phased_ancestry_block_dot(
        _io, j, q, v * weights, out, _n_threads, _buff
    );
}

ADELIE_CORE_MATRIX_NAIVE_SNP_PHASED_ANCESTRY_TP
void
ADELIE_CORE_MATRIX_NAIVE_SNP_PHASED_ANCESTRY::bmul_safe(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
{
    base_t::check_bmul(j, q, v.size(), weights.size(), out.size(), rows(), cols());
    vec_value_t buff(q * _n_threads * (_n_threads > 1) * !util::omp_in_parallel());
    snp_phased_ancestry_block_dot(
        _io, j, q, v * weights, out, _n_threads, buff
    );
}

ADELIE_CORE_MATRIX_NAIVE_SNP_PHASED_ANCESTRY_TP
void
ADELIE_CORE_MATRIX_NAIVE_SNP_PHASED_ANCESTRY::btmul(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    Eigen::Ref<vec_value_t> out
)
{
    base_t::check_btmul(j, q, v.size(), out.size(), rows(), cols());
    snp_phased_ancestry_block_axi(
        _io, j, q, v, out, _n_threads
    );
}

ADELIE_CORE_MATRIX_NAIVE_SNP_PHASED_ANCESTRY_TP
void
ADELIE_CORE_MATRIX_NAIVE_SNP_PHASED_ANCESTRY::mul(
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

ADELIE_CORE_MATRIX_NAIVE_SNP_PHASED_ANCESTRY_TP
void
ADELIE_CORE_MATRIX_NAIVE_SNP_PHASED_ANCESTRY::cov(
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
        const auto snp0 = begin0 / A;
        const auto ancestry_lower0 = begin0 % A;
        const auto ancestry_upper0 = std::min<int>(ancestry_lower0 + q - n_solved0, A);

        int n_solved1 = 0;
        while (n_solved1 <= n_solved0) {
            const auto begin1 = j + n_solved1;
            const auto ancestry_lower1 = begin1 % A;
            const auto ancestry_upper1 = std::min<int>(ancestry_lower1 + q - n_solved1, A);

            if (n_solved0 == n_solved1) {
                const auto snp = snp0;
                const auto a_low = ancestry_lower0;
                const auto a_high = ancestry_upper0;
                const auto a_size = a_high - a_low;

                // increase buffer including cross-term computation part as well
                if (
                    static_cast<size_t>(buff.size()) < a_size * _n_threads &&
                    _n_threads > 1 &&
                    !util::omp_in_parallel()
                ) {
                    buff.resize(a_size * _n_threads);
                }

                // compute quadratic diagonal part
                auto out_diag = out.diagonal().segment(n_solved0, a_size);
                snp_phased_ancestry_block_dot(
                    _io, begin0, a_size, sqrt_weights.square(), out_diag, _n_threads, buff
                );
                
                const auto routine = [&](auto k0, auto k1) {
                    const auto sum = snp_phased_ancestry_cross_dot(
                        _io,
                        snp * A + a_low + k0,
                        snp * A + a_low + k1,
                        sqrt_weights.square()
                    );
                    const auto kk0 = n_solved0 + k0;
                    const auto kk1 = n_solved0 + k1;
                    out(kk0, kk1) += sum * (1 + (kk0 == kk1));
                };
                if (_n_threads <= 1 || util::omp_in_parallel()) {
                    for (int k0 = 0; k0 < static_cast<int>(a_size); ++k0) {
                        for (int k1 = 0; k1 < static_cast<int>(a_size); ++k1) {
                            routine(k0, k1);
                        }
                    }
                } else {
                    #pragma omp parallel for schedule(static) num_threads(_n_threads) collapse(2)
                    for (int k0 = 0; k0 < static_cast<int>(a_size); ++k0) {
                        for (int k1 = 0; k1 < static_cast<int>(a_size); ++k1) {
                            routine(k0, k1);
                        }
                    }
                }

                for (size_t k0 = 0; k0 < a_size; ++k0) {
                    for (size_t k1 = 0; k1 < k0; ++k1) {
                        const auto kk0 = n_solved0 + k0;
                        const auto kk1 = n_solved0 + k1;
                        const auto tmp = out(kk0, kk1);
                        out(kk0, kk1) += out(kk1, kk0);
                        out(kk1, kk0) += tmp;
                    }
                }

                n_solved1 += ancestry_upper1 - ancestry_lower1;
                continue;
            }

            /* general routine */

            const auto ancestry_size0 = ancestry_upper0-ancestry_lower0;
            const auto ancestry_size1 = ancestry_upper1-ancestry_lower1;
            for (size_t a0 = 0; a0 < ancestry_size0; ++a0) {
                size_t nnz = 0;
                for (size_t hap0 = 0; hap0 < io_t::n_haps; ++hap0) {
                    auto it = _io.begin(snp0, ancestry_lower0+a0, hap0);
                    const auto end = _io.end(snp0, ancestry_lower0+a0, hap0);
                    for (; it != end; ++it) {
                        const auto idx = *it;
                        if (!bbuff[idx]) {
                            ibuff[nnz] = idx;
                            ++nnz;
                        }
                        bbuff[idx] += 1;
                    }
                }

                for (size_t a1 = 0; a1 < ancestry_size1; ++a1) {
                    const auto sum = snp_phased_ancestry_dot(
                        _io, begin1 + a1, 
                        vec_value_t::NullaryExpr(sqrt_weights.size(), [&](auto i) {
                            const auto sqrt_wi = sqrt_weights[i];
                            return sqrt_wi * sqrt_wi * bbuff[i];
                        }),
                        _n_threads,
                        buff
                    );
                    const auto kk0 = n_solved0 + a0;
                    const auto kk1 = n_solved1 + a1;
                    out(kk0, kk1) = sum;
                    out(kk1, kk0) = sum;
                }

                for (size_t i = 0; i < nnz; ++i) {
                    bbuff[ibuff[i]] = 0;
                }
            }

            n_solved1 += ancestry_upper1 - ancestry_lower1;
        }
        n_solved0 += ancestry_upper0 - ancestry_lower0;
    }     
}

ADELIE_CORE_MATRIX_NAIVE_SNP_PHASED_ANCESTRY_TP
int
ADELIE_CORE_MATRIX_NAIVE_SNP_PHASED_ANCESTRY::rows() const 
{ 
    return _io.rows(); 
}

ADELIE_CORE_MATRIX_NAIVE_SNP_PHASED_ANCESTRY_TP
int
ADELIE_CORE_MATRIX_NAIVE_SNP_PHASED_ANCESTRY::cols() const 
{ 
    return _io.snps() * ancestries(); 
}

ADELIE_CORE_MATRIX_NAIVE_SNP_PHASED_ANCESTRY_TP
void
ADELIE_CORE_MATRIX_NAIVE_SNP_PHASED_ANCESTRY::sq_mul(
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
{
    const auto routine = [&](int t) {
        out[t] = _sq_cmul(t, weights, out /* unused */);
    };
    util::omp_parallel_for(routine, 0, cols(), _n_threads);
}

ADELIE_CORE_MATRIX_NAIVE_SNP_PHASED_ANCESTRY_TP
void
ADELIE_CORE_MATRIX_NAIVE_SNP_PHASED_ANCESTRY::sp_tmul(
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

ADELIE_CORE_MATRIX_NAIVE_SNP_PHASED_ANCESTRY_TP
void
ADELIE_CORE_MATRIX_NAIVE_SNP_PHASED_ANCESTRY::mean(
    const Eigen::Ref<const vec_value_t>&,
    Eigen::Ref<vec_value_t> out
) const
{
    out.setZero();
}

ADELIE_CORE_MATRIX_NAIVE_SNP_PHASED_ANCESTRY_TP
void
ADELIE_CORE_MATRIX_NAIVE_SNP_PHASED_ANCESTRY::var(
    const Eigen::Ref<const vec_value_t>&,
    const Eigen::Ref<const vec_value_t>&,
    Eigen::Ref<vec_value_t> out
) const
{
    out.setOnes();
}

} // namespace matrix
} // namespace adelie_core