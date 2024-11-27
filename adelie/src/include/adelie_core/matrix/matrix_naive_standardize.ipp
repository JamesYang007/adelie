#pragma once
#include <adelie_core/matrix/matrix_naive_standardize.hpp>
#include <adelie_core/matrix/utils.hpp>

namespace adelie_core {
namespace matrix {

ADELIE_CORE_MATRIX_NAIVE_STANDARDIZE_TP
ADELIE_CORE_MATRIX_NAIVE_STANDARDIZE::MatrixNaiveStandardize(
    base_t& mat,
    const Eigen::Ref<const vec_value_t>& centers,
    const Eigen::Ref<const vec_value_t>& scales,
    size_t n_threads
):
    _mat(&mat),
    _centers(centers.data(), centers.size()),
    _scales(scales.data(), scales.size()),
    _n_threads(n_threads),
    _buff(mat.cols() + n_threads)
{
    const auto p = mat.cols();

    if (centers.size() != p) {
        throw util::adelie_core_error(
            "centers must be (p,) where mat is (n, p)."
        );
    }
    if (scales.size() != p) {
        throw util::adelie_core_error(
            "scales must be (p,) where mat is (n, p)."
        );
    }
    if (n_threads < 1) {
        throw util::adelie_core_error("n_threads must be >= 1.");
    }
}

ADELIE_CORE_MATRIX_NAIVE_STANDARDIZE_TP
typename ADELIE_CORE_MATRIX_NAIVE_STANDARDIZE::value_t
ADELIE_CORE_MATRIX_NAIVE_STANDARDIZE::cmul(
    int j, 
    const Eigen::Ref<const vec_value_t>& v,
    const Eigen::Ref<const vec_value_t>& weights
)
{
    base_t::check_cmul(j, v.size(), weights.size(), rows(), cols());
    const auto c = _centers[j];
    const auto vw_sum = (
        (c == 0) ? 0 : ddot(v.matrix(), weights.matrix(), _n_threads, _buff)
    );
    return (_mat->cmul(j, v, weights) - c * vw_sum) / _scales[j];
}

ADELIE_CORE_MATRIX_NAIVE_STANDARDIZE_TP
typename ADELIE_CORE_MATRIX_NAIVE_STANDARDIZE::value_t
ADELIE_CORE_MATRIX_NAIVE_STANDARDIZE::cmul_safe(
    int j, 
    const Eigen::Ref<const vec_value_t>& v,
    const Eigen::Ref<const vec_value_t>& weights
) const
{
    base_t::check_cmul(j, v.size(), weights.size(), rows(), cols());
    vec_value_t buff(_n_threads * (_n_threads > 1) * !util::omp_in_parallel());
    const auto c = _centers[j];
    const auto vw_sum = (
        (c == 0) ? 0 : ddot(v.matrix(), weights.matrix(), _n_threads, buff)
    );
    return (_mat->cmul_safe(j, v, weights) - c * vw_sum) / _scales[j];
}

ADELIE_CORE_MATRIX_NAIVE_STANDARDIZE_TP
void
ADELIE_CORE_MATRIX_NAIVE_STANDARDIZE::ctmul(
    int j, 
    value_t v, 
    Eigen::Ref<vec_value_t> out
)
{
    base_t::check_ctmul(j, out.size(), rows(), cols());
    const auto vs = v / _scales[j];
    _mat->ctmul(j, vs, out);
    const auto vsc = _centers[j] * vs;
    if (!vsc) return;
    dvsubi(
        out, 
        vec_value_t::NullaryExpr(out.size(), [&](auto) {
            return vsc;
        }),
        _n_threads
    );
}

ADELIE_CORE_MATRIX_NAIVE_STANDARDIZE_TP
void
ADELIE_CORE_MATRIX_NAIVE_STANDARDIZE::bmul(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
)
{
    base_t::check_bmul(j, q, v.size(), weights.size(), out.size(), rows(), cols());
    _mat->bmul(j, q, v, weights, out);
    const auto c = _centers.segment(j, q);
    const auto vw_sum = (
        (c == 0).all() ? 0 : ddot(v.matrix(), weights.matrix(), _n_threads, _buff)
    );
    const auto s = _scales.segment(j, q);
    dvveq(out, (out - vw_sum * c) / s, _n_threads);
}

ADELIE_CORE_MATRIX_NAIVE_STANDARDIZE_TP
void
ADELIE_CORE_MATRIX_NAIVE_STANDARDIZE::bmul_safe(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
{
    base_t::check_bmul(j, q, v.size(), weights.size(), out.size(), rows(), cols());
    vec_value_t buff(_n_threads * (_n_threads > 1) * !util::omp_in_parallel());
    _mat->bmul_safe(j, q, v, weights, out);
    const auto c = _centers.segment(j, q);
    const auto vw_sum = (
        (c == 0).all() ? 0 : ddot(v.matrix(), weights.matrix(), _n_threads, buff)
    );
    const auto s = _scales.segment(j, q);
    dvveq(out, (out - vw_sum * c) / s, _n_threads);
}

ADELIE_CORE_MATRIX_NAIVE_STANDARDIZE_TP
void
ADELIE_CORE_MATRIX_NAIVE_STANDARDIZE::btmul(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    Eigen::Ref<vec_value_t> out
)
{
    base_t::check_btmul(j, q, v.size(), out.size(), rows(), cols());
    auto vs = _buff.head(q);
    const auto s = _scales.segment(j, q);
    dvveq(vs, v / s, _n_threads);
    _mat->btmul(j, q, vs, out);

    auto buff = _buff.segment(q, _n_threads);
    const auto vsc = ddot(
        _centers.segment(j, q).matrix(),
        vs.matrix(),
        _n_threads,
        buff
    );
    if (!vsc) return;
    dvsubi(
        out, 
        vec_value_t::NullaryExpr(out.size(), [&](auto) { return vsc; }),
        _n_threads
    );
}

ADELIE_CORE_MATRIX_NAIVE_STANDARDIZE_TP
void
ADELIE_CORE_MATRIX_NAIVE_STANDARDIZE::mul(
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
{
    vec_value_t buff(_n_threads * (_n_threads > 1) * !util::omp_in_parallel());
    _mat->mul(v, weights, out);
    const auto vw_sum = ddot(v.matrix(), weights.matrix(), _n_threads, buff);
    dvveq(out, (out - vw_sum * _centers) / _scales, _n_threads);
}

ADELIE_CORE_MATRIX_NAIVE_STANDARDIZE_TP
int
ADELIE_CORE_MATRIX_NAIVE_STANDARDIZE::rows() const
{
    return _mat->rows();
}

ADELIE_CORE_MATRIX_NAIVE_STANDARDIZE_TP
int
ADELIE_CORE_MATRIX_NAIVE_STANDARDIZE::cols() const
{
    return _mat->cols();
}

ADELIE_CORE_MATRIX_NAIVE_STANDARDIZE_TP
void
ADELIE_CORE_MATRIX_NAIVE_STANDARDIZE::cov(
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

    _mat->cov(j, q, sqrt_weights, out);

    const auto centers = _centers.segment(j, q);
    const auto scales = _scales.segment(j, q);

    if ((centers != 0).any()) {
        vec_value_t means(q);
        auto out_lower = out.template selfadjointView<Eigen::Lower>();
        _mat->bmul_safe(j, q, sqrt_weights, sqrt_weights, means);
        out_lower.rankUpdate(centers.matrix().transpose(), means.matrix().transpose(), -1);
        out_lower.rankUpdate(centers.matrix().transpose(), sqrt_weights.square().sum());
        out.template triangularView<Eigen::Upper>() = out.transpose();
    }

    out.array().rowwise() /= scales;
    out.array().colwise() /= scales.matrix().transpose().array();
}

ADELIE_CORE_MATRIX_NAIVE_STANDARDIZE_TP
void
ADELIE_CORE_MATRIX_NAIVE_STANDARDIZE::sq_mul(
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
{
    _mat->sq_mul(weights, out);
    vec_value_t mat_means(out.size());
    vec_value_t ones = vec_value_t::Ones(weights.size());
    _mat->mul(ones, weights, mat_means);
    const auto w_sum = weights.sum();
    dvveq(out, (out - 2 * _centers * mat_means + w_sum * _centers.square()) / _scales.square(), _n_threads);
}

ADELIE_CORE_MATRIX_NAIVE_STANDARDIZE_TP
void
ADELIE_CORE_MATRIX_NAIVE_STANDARDIZE::sp_tmul(
    const sp_mat_value_t& v, 
    Eigen::Ref<rowmat_value_t> out
) const
{
    base_t::check_sp_tmul(
        v.rows(), v.cols(), out.rows(), out.cols(), rows(), cols()
    );
    sp_mat_value_t vs = v;
    if (!vs.isCompressed()) vs.makeCompressed();
    for (int k = 0; k < vs.outerSize(); ++k) {
        const auto outer = vs.outerIndexPtr();
        const auto inner = vs.innerIndexPtr() + outer[k];
        const auto value = vs.valuePtr() + outer[k];
        const auto size = outer[k+1] - outer[k];
        for (int i = 0; i < size; ++i) {
            value[i] /= _scales[inner[i]];
        }
    }
    _mat->sp_tmul(vs, out);

    const auto routine = [&](auto k) {
        typename sp_mat_value_t::InnerIterator it(vs, k);
        auto out_k = out.row(k);
        value_t vsc = 0;
        for (; it; ++it) {
            const auto idx = it.index();
            vsc += it.value() * _centers[idx];
        }
        if (vsc) out_k.array() -= vsc;
    };
    util::omp_parallel_for(routine, 0, v.outerSize(), _n_threads);
}

ADELIE_CORE_MATRIX_NAIVE_STANDARDIZE_TP
void
ADELIE_CORE_MATRIX_NAIVE_STANDARDIZE::mean(
    const Eigen::Ref<const vec_value_t>&,
    Eigen::Ref<vec_value_t> out
) const
{
    out.setZero();
}

ADELIE_CORE_MATRIX_NAIVE_STANDARDIZE_TP
void
ADELIE_CORE_MATRIX_NAIVE_STANDARDIZE::var(
    const Eigen::Ref<const vec_value_t>&,
    const Eigen::Ref<const vec_value_t>&,
    Eigen::Ref<vec_value_t> out
) const
{
    out.setOnes();
}

} // namespace matrix
} // namespace adelie_core 