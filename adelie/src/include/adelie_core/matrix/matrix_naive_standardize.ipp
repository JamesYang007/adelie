#pragma once
#include <adelie_core/matrix/matrix_naive_standardize.hpp>
#include <adelie_core/matrix/utils.hpp>

namespace adelie_core {
namespace matrix {

template <class ValueType, class IndexType>
MatrixNaiveStandardize<ValueType, IndexType>::MatrixNaiveStandardize(
    base_t& mat,
    const Eigen::Ref<const vec_value_t>& centers,
    const Eigen::Ref<const vec_value_t>& scales,
    size_t n_threads
):
    _mat(&mat),
    _centers(centers.data(), centers.size()),
    _scales(scales.data(), scales.size()),
    _n_threads(n_threads),
    _buff(std::max<size_t>(mat.cols(), n_threads))
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

template <class ValueType, class IndexType>
typename MatrixNaiveStandardize<ValueType, IndexType>::value_t
MatrixNaiveStandardize<ValueType, IndexType>::cmul(
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

template <class ValueType, class IndexType>
void
MatrixNaiveStandardize<ValueType, IndexType>::ctmul(
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

template <class ValueType, class IndexType>
void
MatrixNaiveStandardize<ValueType, IndexType>::bmul(
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

template <class ValueType, class IndexType>
void
MatrixNaiveStandardize<ValueType, IndexType>::btmul(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    Eigen::Ref<vec_value_t> out
)
{
    base_t::check_btmul(j, q, v.size(), out.size(), rows(), cols());
    auto vs = _buff.segment(0, q);
    const auto s = _scales.segment(j, q);
    dvveq(vs, v / s, _n_threads);
    _mat->btmul(j, q, vs, out);

    const auto vsc = ddot(
        _centers.segment(j, q).matrix(),
        vs.matrix(),
        _n_threads,
        _buff
    );
    if (!vsc) return;
    dvsubi(
        out, 
        vec_value_t::NullaryExpr(out.size(), [&](auto) { return vsc; }),
        _n_threads
    );
}

template <class ValueType, class IndexType>
void
MatrixNaiveStandardize<ValueType, IndexType>::mul(
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
)
{
    _mat->mul(v, weights, out);
    const auto vw_sum = ddot(v.matrix(), weights.matrix(), _n_threads, _buff);
    dvveq(out, (out - vw_sum * _centers) / _scales, _n_threads);
}

template <class ValueType, class IndexType>
int
MatrixNaiveStandardize<ValueType, IndexType>::rows() const
{
    return _mat->rows();
}

template <class ValueType, class IndexType>
int
MatrixNaiveStandardize<ValueType, IndexType>::cols() const
{
    return _mat->cols();
}

template <class ValueType, class IndexType>
void
MatrixNaiveStandardize<ValueType, IndexType>::cov(
    int j, int q,
    const Eigen::Ref<const vec_value_t>& sqrt_weights,
    Eigen::Ref<colmat_value_t> out,
    Eigen::Ref<colmat_value_t> buffer
)
{
    base_t::check_cov(
        j, q, sqrt_weights.size(), 
        out.rows(), out.cols(), buffer.rows(), buffer.cols(), 
        rows(), cols()
    );

    _mat->cov(j, q, sqrt_weights, out, buffer);

    const auto centers = _centers.segment(j, q);
    const auto scales = _scales.segment(j, q);

    if ((centers != 0).any()) {
        auto out_lower = out.template selfadjointView<Eigen::Lower>();
        auto means = _buff.segment(j, q);
        _mat->bmul(j, q, sqrt_weights, sqrt_weights, means);
        out_lower.rankUpdate(centers.matrix().transpose(), means.matrix().transpose(), -1);
        out_lower.rankUpdate(centers.matrix().transpose(), sqrt_weights.square().sum());
        out.template triangularView<Eigen::Upper>() = out.transpose();
    }

    out.array().rowwise() /= scales;
    out.array().colwise() /= scales.matrix().transpose().array();
}

template <class ValueType, class IndexType>
void
MatrixNaiveStandardize<ValueType, IndexType>::sp_tmul(
    const sp_mat_value_t& v, 
    Eigen::Ref<rowmat_value_t> out
)
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
    if (_n_threads <= 1) {
        for (int k = 0; k < v.outerSize(); ++k) routine(k);
    } else {
        #pragma omp parallel for schedule(static) num_threads(_n_threads)
        for (int k = 0; k < v.outerSize(); ++k) routine(k);
    }
}

} // namespace matrix
} // namespace adelie_core 