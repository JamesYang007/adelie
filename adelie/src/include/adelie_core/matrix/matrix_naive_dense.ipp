#pragma once
#include <adelie_core/matrix/matrix_naive_dense.hpp>
#include <adelie_core/matrix/utils.hpp>

namespace adelie_core {
namespace matrix {

ADELIE_CORE_MATRIX_NAIVE_DENSE_TP
ADELIE_CORE_MATRIX_NAIVE_DENSE::MatrixNaiveDense(
    const Eigen::Ref<const dense_t>& mat,
    size_t n_threads
): 
    _mat(mat.data(), mat.rows(), mat.cols()),
    _n_threads(n_threads),
    _buff(_n_threads, std::min(mat.rows(), mat.cols())),
    _vbuff(mat.rows())
{
    if (n_threads < 1) {
        throw util::adelie_core_error("n_threads must be >= 1.");
    }
}

ADELIE_CORE_MATRIX_NAIVE_DENSE_TP
typename ADELIE_CORE_MATRIX_NAIVE_DENSE::value_t
ADELIE_CORE_MATRIX_NAIVE_DENSE::cmul(
    int j, 
    const Eigen::Ref<const vec_value_t>& v,
    const Eigen::Ref<const vec_value_t>& weights
) 
{
    base_t::check_cmul(j, v.size(), weights.size(), rows(), cols());
    Eigen::Map<vec_value_t> vbuff(_buff.data(), _n_threads);
    return ddot(_mat.col(j), (v * weights).matrix(), _n_threads, vbuff);
}

ADELIE_CORE_MATRIX_NAIVE_DENSE_TP
typename ADELIE_CORE_MATRIX_NAIVE_DENSE::value_t
ADELIE_CORE_MATRIX_NAIVE_DENSE::cmul_safe(
    int j, 
    const Eigen::Ref<const vec_value_t>& v,
    const Eigen::Ref<const vec_value_t>& weights
) const
{
    base_t::check_cmul(j, v.size(), weights.size(), rows(), cols());
    vec_value_t vbuff(_n_threads * (_n_threads > 1) * !util::omp_in_parallel());
    return ddot(_mat.col(j), (v * weights).matrix(), _n_threads, vbuff);
}

ADELIE_CORE_MATRIX_NAIVE_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_DENSE::ctmul(
    int j, 
    value_t v, 
    Eigen::Ref<vec_value_t> out
)
{
    base_t::check_ctmul(j, out.size(), rows(), cols());
    dvaddi(out, v * _mat.col(j).transpose().array(), _n_threads);
}

ADELIE_CORE_MATRIX_NAIVE_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_DENSE::bmul(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
)
{
    base_t::check_bmul(j, q, v.size(), weights.size(), out.size(), rows(), cols());
    auto outm = out.matrix();
    dvveq(_vbuff, v * weights, _n_threads);
    dgemv(
        _mat.middleCols(j, q),
        _vbuff.matrix(),
        _n_threads,
        _buff,
        outm
    );
}

ADELIE_CORE_MATRIX_NAIVE_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_DENSE::bmul_safe(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
{
    base_t::check_bmul(j, q, v.size(), weights.size(), out.size(), rows(), cols());
    const auto n = _mat.rows();
    auto out_m = out.matrix();
    vec_value_t vbuff(n);
    rowmat_value_t buff(_n_threads * (_n_threads > 1) * !util::omp_in_parallel(), q * (n > q));
    dvveq(vbuff, v * weights, _n_threads);
    dgemv(
        _mat.middleCols(j, q),
        vbuff.matrix(),
        _n_threads,
        buff,
        out_m
    );
}

ADELIE_CORE_MATRIX_NAIVE_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_DENSE::btmul(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    Eigen::Ref<vec_value_t> out
)
{
    base_t::check_btmul(j, q, v.size(), out.size(), rows(), cols());
    auto outm = out.matrix();
    dgemv<util::operator_type::_add>(
        _mat.middleCols(j, q).transpose(),
        v.matrix(),
        _n_threads,
        _buff,
        outm
    );
}

ADELIE_CORE_MATRIX_NAIVE_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_DENSE::mul(
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
{
    const auto n = _mat.rows();
    const auto p = _mat.cols();
    auto out_m = out.matrix();
    vec_value_t vbuff(n);
    rowmat_value_t buff(_n_threads * (_n_threads > 1) * !util::omp_in_parallel(), p * (n > p));
    dvveq(vbuff, v * weights, _n_threads);
    dgemv(
        _mat,
        vbuff.matrix(),
        _n_threads,
        buff,
        out_m
    );
}

ADELIE_CORE_MATRIX_NAIVE_DENSE_TP
int
ADELIE_CORE_MATRIX_NAIVE_DENSE::rows() const
{
    return _mat.rows();
}

ADELIE_CORE_MATRIX_NAIVE_DENSE_TP
int
ADELIE_CORE_MATRIX_NAIVE_DENSE::cols() const
{
    return _mat.cols();
}

ADELIE_CORE_MATRIX_NAIVE_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_DENSE::cov(
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
    
    if (q == 1) {
        const auto sqrt_w_mj = (_mat.col(j).transpose().array() * sqrt_weights).matrix();
        vec_value_t vbuff(_n_threads * (_n_threads > 1) * !util::omp_in_parallel());
        out(0, 0) = ddot(sqrt_w_mj, sqrt_w_mj, _n_threads, vbuff);
        return;
    }

    const auto n = _mat.rows();
    colmat_value_t Xj(n, q);
    
    auto Xj_array = Xj.array();
    dmmeq(
        Xj_array, 
        _mat.middleCols(j, q).array().colwise() * sqrt_weights.matrix().transpose().array(),
        _n_threads
    );

    out.setZero();
    auto out_lower = out.template selfadjointView<Eigen::Lower>();
    out_lower.rankUpdate(Xj.transpose());
    out.template triangularView<Eigen::Upper>() = out.transpose();
}

ADELIE_CORE_MATRIX_NAIVE_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_DENSE::sq_mul(
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
{
    const auto n = _mat.rows();
    const auto p = _mat.cols();
    auto out_m = out.matrix();
    rowmat_value_t buff(_n_threads * (_n_threads > 1) * !util::omp_in_parallel(), p * (n > p));
    dgemv(
        _mat.array().square().matrix(),
        weights.matrix(),
        _n_threads,
        buff,
        out_m
    );
}

ADELIE_CORE_MATRIX_NAIVE_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_DENSE::sp_tmul(
    const sp_mat_value_t& v, 
    Eigen::Ref<rowmat_value_t> out
) const
{
    base_t::check_sp_tmul(
        v.rows(), v.cols(), out.rows(), out.cols(), rows(), cols()
    );
    if (_n_threads <= 1) {
        out.noalias() = v * _mat.transpose();
        return;
    }
    sp_mat_value_t vc;
    if (!v.isCompressed()) {
        vc = v;
        if (!vc.isCompressed()) vc.makeCompressed();
    }
    const sp_mat_value_t& v_ref = (vc.size() != 0) ? vc : v;

    const auto outer = v_ref.outerIndexPtr();
    const auto inner = v_ref.innerIndexPtr();
    const auto value = v_ref.valuePtr();
    const auto routine = [&](auto k) {
        const Eigen::Map<const sp_mat_value_t> vk(
            1,
            v_ref.cols(),
            outer[k+1] - outer[k],
            outer + k,
            inner,
            value
        );
        auto out_k = out.row(k);
        out_k = vk * _mat.transpose();
    };
    util::omp_parallel_for(routine, 0, v_ref.outerSize(), _n_threads);
}

} // namespace matrix
} // namespace adelie_core