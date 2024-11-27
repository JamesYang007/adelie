#pragma once
#include <adelie_core/matrix/matrix_naive_snp_unphased.hpp>
#include <adelie_core/matrix/utils.hpp>

namespace adelie_core {
namespace matrix {

ADELIE_CORE_MATRIX_NAIVE_SNP_UNPHASED_TP
typename ADELIE_CORE_MATRIX_NAIVE_SNP_UNPHASED::value_t
ADELIE_CORE_MATRIX_NAIVE_SNP_UNPHASED::_cmul(
    int j, 
    const Eigen::Ref<const vec_value_t>& v,
    const Eigen::Ref<const vec_value_t>& weights,
    size_t n_threads,
    Eigen::Ref<vec_value_t> buff
) const 
{
    return snp_unphased_dot(
        [](auto x) { return x; },
        _io, j, v * weights, n_threads, buff
    );
}

ADELIE_CORE_MATRIX_NAIVE_SNP_UNPHASED_TP
typename ADELIE_CORE_MATRIX_NAIVE_SNP_UNPHASED::value_t
ADELIE_CORE_MATRIX_NAIVE_SNP_UNPHASED::_sq_cmul(
    int j, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> buff
) const
{
    constexpr size_t n_threads = 1;
    return snp_unphased_dot(
        [](auto x) { return x * x; },
        _io, j, weights, n_threads, buff
    );
}

ADELIE_CORE_MATRIX_NAIVE_SNP_UNPHASED_TP
void
ADELIE_CORE_MATRIX_NAIVE_SNP_UNPHASED::_ctmul(
    int j, 
    value_t v, 
    Eigen::Ref<vec_value_t> out,
    size_t n_threads
) const
{
    snp_unphased_axi(_io, j, v, out, n_threads);
}

ADELIE_CORE_MATRIX_NAIVE_SNP_UNPHASED_TP
ADELIE_CORE_MATRIX_NAIVE_SNP_UNPHASED::MatrixNaiveSNPUnphased(
    const io_t& io,
    size_t n_threads
): 
    _io(io),
    _n_threads(n_threads),
    _buff(n_threads)
{
    if (n_threads < 1) {
        throw util::adelie_core_error("n_threads must be >= 1.");
    }
}

ADELIE_CORE_MATRIX_NAIVE_SNP_UNPHASED_TP
typename ADELIE_CORE_MATRIX_NAIVE_SNP_UNPHASED::value_t
ADELIE_CORE_MATRIX_NAIVE_SNP_UNPHASED::cmul(
    int j, 
    const Eigen::Ref<const vec_value_t>& v,
    const Eigen::Ref<const vec_value_t>& weights
)
{
    base_t::check_cmul(j, v.size(), weights.size(), rows(), cols());
    return _cmul(j, v, weights, _n_threads, _buff);
}

ADELIE_CORE_MATRIX_NAIVE_SNP_UNPHASED_TP
typename ADELIE_CORE_MATRIX_NAIVE_SNP_UNPHASED::value_t
ADELIE_CORE_MATRIX_NAIVE_SNP_UNPHASED::cmul_safe(
    int j, 
    const Eigen::Ref<const vec_value_t>& v,
    const Eigen::Ref<const vec_value_t>& weights
) const
{
    base_t::check_cmul(j, v.size(), weights.size(), rows(), cols());
    vec_value_t buff(_n_threads * (_n_threads > 1) * !util::omp_in_parallel());
    return _cmul(j, v, weights, _n_threads, buff);
}

ADELIE_CORE_MATRIX_NAIVE_SNP_UNPHASED_TP
void
ADELIE_CORE_MATRIX_NAIVE_SNP_UNPHASED::ctmul(
    int j, 
    value_t v, 
    Eigen::Ref<vec_value_t> out
)
{
    base_t::check_ctmul(j, out.size(), rows(), cols());
    _ctmul(j, v, out, _n_threads);
}

ADELIE_CORE_MATRIX_NAIVE_SNP_UNPHASED_TP
void
ADELIE_CORE_MATRIX_NAIVE_SNP_UNPHASED::bmul(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
)
{
    base_t::check_bmul(j, q, v.size(), weights.size(), out.size(), rows(), cols());
    for (int t = 0; t < q; ++t) {
        out[t] = _cmul(j + t, v, weights, _n_threads, _buff);
    }
}

ADELIE_CORE_MATRIX_NAIVE_SNP_UNPHASED_TP
void
ADELIE_CORE_MATRIX_NAIVE_SNP_UNPHASED::bmul_safe(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
{
    base_t::check_bmul(j, q, v.size(), weights.size(), out.size(), rows(), cols());
    vec_value_t buff(_n_threads * (_n_threads > 1) * !util::omp_in_parallel());
    for (int t = 0; t < q; ++t) {
        out[t] = _cmul(j + t, v, weights, _n_threads, buff);
    }
}

ADELIE_CORE_MATRIX_NAIVE_SNP_UNPHASED_TP
void
ADELIE_CORE_MATRIX_NAIVE_SNP_UNPHASED::btmul(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    Eigen::Ref<vec_value_t> out
)
{
    base_t::check_btmul(j, q, v.size(), out.size(), rows(), cols());
    for (int t = 0; t < q; ++t) {
        _ctmul(j + t, v[t], out, _n_threads);
    }
}

ADELIE_CORE_MATRIX_NAIVE_SNP_UNPHASED_TP
void
ADELIE_CORE_MATRIX_NAIVE_SNP_UNPHASED::mul(
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

ADELIE_CORE_MATRIX_NAIVE_SNP_UNPHASED_TP
void
ADELIE_CORE_MATRIX_NAIVE_SNP_UNPHASED::cov(
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

    vec_index_t ibuff(_io.rows());
    vec_value_t vbuff(_io.rows());
    vec_value_t buff(_n_threads * (_n_threads > 1) * !util::omp_in_parallel());
    vbuff.setConstant(_max);

    for (int i1 = 0; i1 < q; ++i1) 
    {
        const auto index_1 = j+i1;
        const value_t imp_1 = _io.impute()[index_1];

        // if covariance must be computed,
        // cache index_1 information. 
        size_t nnz = 0;
        if (i1) {
            for (size_t c = 0; c < io_t::n_categories; ++c) {
                auto it = _io.begin(index_1, c);
                const auto end = _io.end(index_1, c);
                const value_t val = (c == 0) ? imp_1 : c;
                for (; it != end; ++it) {
                    const auto idx = *it;
                    vbuff[idx] = val;
                    ibuff[nnz] = idx;
                    ++nnz;
                }
            }
        } 

        for (int i2 = 0; i2 <= i1; ++i2) {
            if (i1 == i2) {
                out(i1, i1) = snp_unphased_dot(
                    [](auto x) { return x * x; },
                    _io, 
                    index_1, 
                    sqrt_weights.square(), 
                    _n_threads, 
                    buff
                );
                continue;
            }
            const auto index_2 = j+i2;
            out(i1, i2) = snp_unphased_dot(
                [](auto x) { return x; },
                _io, 
                index_2,
                sqrt_weights.square() * (
                    (vbuff != _max).template cast<value_t>() * vbuff
                ),
                _n_threads,
                buff
            );
        }

        // keep invariance by populating with inf
        for (size_t i = 0; i < nnz; ++i) {
            vbuff[ibuff[i]] = _max;
        }
    }

    for (int i1 = 0; i1 < q; ++i1) {
        for (int i2 = i1+1; i2 < q; ++i2) {
            out(i1, i2) = out(i2, i1);
        }
    }
}

ADELIE_CORE_MATRIX_NAIVE_SNP_UNPHASED_TP
int
ADELIE_CORE_MATRIX_NAIVE_SNP_UNPHASED::rows() const 
{ 
    return _io.rows(); 
}

ADELIE_CORE_MATRIX_NAIVE_SNP_UNPHASED_TP
int
ADELIE_CORE_MATRIX_NAIVE_SNP_UNPHASED::cols() const 
{ 
    return _io.cols(); 
}

ADELIE_CORE_MATRIX_NAIVE_SNP_UNPHASED_TP
void
ADELIE_CORE_MATRIX_NAIVE_SNP_UNPHASED::sq_mul(
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
{
    const auto routine = [&](int t) {
        out[t] = _sq_cmul(t, weights, out /* unused */);
    };
    util::omp_parallel_for(routine, 0, cols(), _n_threads);
}

ADELIE_CORE_MATRIX_NAIVE_SNP_UNPHASED_TP
void
ADELIE_CORE_MATRIX_NAIVE_SNP_UNPHASED::sp_tmul(
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

ADELIE_CORE_MATRIX_NAIVE_SNP_UNPHASED_TP
void
ADELIE_CORE_MATRIX_NAIVE_SNP_UNPHASED::mean(
    const Eigen::Ref<const vec_value_t>&,
    Eigen::Ref<vec_value_t> out
) const
{
    out.setZero();
}

ADELIE_CORE_MATRIX_NAIVE_SNP_UNPHASED_TP
void
ADELIE_CORE_MATRIX_NAIVE_SNP_UNPHASED::var(
    const Eigen::Ref<const vec_value_t>&,
    const Eigen::Ref<const vec_value_t>&,
    Eigen::Ref<vec_value_t> out
) const
{
    out.setOnes();
}

} // namespace matrix
} // namespace adelie_core