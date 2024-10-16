#pragma once
#include <adelie_core/matrix/matrix_naive_dense.hpp>
#include <adelie_core/matrix/utils.hpp>

namespace adelie_core {
namespace matrix {

template <class DenseType, class IndexType>
MatrixNaiveDense<DenseType, IndexType>::MatrixNaiveDense(
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

template <class DenseType, class IndexType>
typename MatrixNaiveDense<DenseType, IndexType>::value_t
MatrixNaiveDense<DenseType, IndexType>::cmul(
    int j, 
    const Eigen::Ref<const vec_value_t>& v,
    const Eigen::Ref<const vec_value_t>& weights
) 
{
    base_t::check_cmul(j, v.size(), weights.size(), rows(), cols());
    Eigen::Map<vec_value_t> vbuff(_buff.data(), _n_threads);
    return ddot(_mat.col(j), (v * weights).matrix(), _n_threads, vbuff);
}

template <class DenseType, class IndexType>
void
MatrixNaiveDense<DenseType, IndexType>::ctmul(
    int j, 
    value_t v, 
    Eigen::Ref<vec_value_t> out
)
{
    base_t::check_ctmul(j, out.size(), rows(), cols());
    dvaddi(out, v * _mat.col(j).transpose().array(), _n_threads);
}

template <class DenseType, class IndexType>
void
MatrixNaiveDense<DenseType, IndexType>::bmul(
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

template <class DenseType, class IndexType>
void
MatrixNaiveDense<DenseType, IndexType>::btmul(
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

template <class DenseType, class IndexType>
void
MatrixNaiveDense<DenseType, IndexType>::mul(
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
)
{
    auto outm = out.matrix();
    dvveq(_vbuff, v * weights, _n_threads);
    dgemv(
        _mat,
        _vbuff.matrix(),
        _n_threads,
        _buff,
        outm
    );
}

template <class DenseType, class IndexType>
int
MatrixNaiveDense<DenseType, IndexType>::rows() const
{
    return _mat.rows();
}

template <class DenseType, class IndexType>
int
MatrixNaiveDense<DenseType, IndexType>::cols() const
{
    return _mat.cols();
}

template <class DenseType, class IndexType>
void
MatrixNaiveDense<DenseType, IndexType>::cov(
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
    
    if (q == 1) {
        const auto sqrt_w_mj = (_mat.col(j).transpose().array() * sqrt_weights).matrix();
        Eigen::Map<vec_value_t> vbuff(_buff.data(), _n_threads);
        out(0, 0) = ddot(sqrt_w_mj, sqrt_w_mj, _n_threads, vbuff);
        return;
    }

    auto& Xj = buffer;
    
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

template <class DenseType, class IndexType>
void
MatrixNaiveDense<DenseType, IndexType>::sp_tmul(
    const sp_mat_value_t& v, 
    Eigen::Ref<rowmat_value_t> out
)
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
    #pragma omp parallel for schedule(static) num_threads(_n_threads)
    for (int k = 0; k < v_ref.outerSize(); ++k) {
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
}

} // namespace matrix
} // namespace adelie_core