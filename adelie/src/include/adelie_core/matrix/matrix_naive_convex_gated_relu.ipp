#pragma once
#include <adelie_core/matrix/matrix_naive_convex_gated_relu.hpp>
#include <adelie_core/matrix/utils.hpp>
#include <adelie_core/util/stopwatch.hpp>
#include <adelie_core/util/macros.hpp>

namespace adelie_core {
namespace matrix {

ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_DENSE_TP
auto
ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_DENSE::_cmul(
    int j, 
    const Eigen::Ref<const vec_value_t>& v,
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> buff
) const 
{
    const auto d = _mat.cols();
    const auto j_m = j / d;
    j -= j_m * d;
    const auto j_d = j;
    return ddot(
        _mat.col(j_d).cwiseProduct(_mask.col(j_m).template cast<value_t>()),
        (v * weights).matrix(),
        _n_threads,
        buff
    );
}

ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_DENSE::_ctmul(
    int j, 
    value_t v, 
    Eigen::Ref<vec_value_t> out,
    size_t n_threads
) const
{
    const auto d = _mat.cols();
    const auto j_m = j / d;
    j -= j_m * d;
    const auto j_d = j;
    dvaddi(
        out, 
        v * _mat.col(j_d).cwiseProduct(
            _mask.col(j_m).template cast<value_t>()
        ).array(),
        n_threads
    );
}

ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_DENSE::_bmul(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out,
    Eigen::Ref<vec_value_t> buffer
) const
{
    const auto d = _mat.cols();
    Eigen::Map<rowmat_value_t> buff(buffer.data(), _n_threads, d);
    int n_processed = 0;
    while (n_processed < q) {
        auto k = j + n_processed;
        const auto k_m = k / d;
        k -= k_m * d;
        const auto k_d = k;
        const auto size = std::min<int>(d-k_d, q-n_processed);
        auto out_m = out.segment(n_processed, size).matrix();
        dgemv(
            _mat.middleCols(k_d, size),
            _mask.col(k_m).transpose().template cast<value_t>().cwiseProduct((v * weights).matrix()),
            _n_threads,
            buff,
            out_m
        );
        n_processed += size;
    }
}

ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_DENSE_TP
ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_DENSE::MatrixNaiveConvexGatedReluDense(
    const Eigen::Ref<const dense_t>& mat,
    const Eigen::Ref<const mask_t>& mask,
    size_t n_threads
):
    _mat(mat.data(), mat.rows(), mat.cols()),
    _mask(mask.data(), mask.rows(), mask.cols()),
    _n_threads(n_threads),
    _buff(n_threads * std::min<size_t>(mat.rows(), mat.cols()) + mat.rows())
{
    const auto n = mat.rows();

    if (mask.rows() != n) {
        throw util::adelie_core_error("mask must be (n, m) where mat is (n, d).");
    }
    if (n_threads < 1) {
        throw util::adelie_core_error("n_threads must be >= 1.");
    }
}

ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_DENSE_TP
typename ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_DENSE::value_t
ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_DENSE::cmul(
    int j, 
    const Eigen::Ref<const vec_value_t>& v,
    const Eigen::Ref<const vec_value_t>& weights
) 
{
    base_t::check_cmul(j, v.size(), weights.size(), rows(), cols());
    return _cmul(j, v, weights, _buff);
}

ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_DENSE_TP
typename ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_DENSE::value_t
ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_DENSE::cmul_safe(
    int j, 
    const Eigen::Ref<const vec_value_t>& v,
    const Eigen::Ref<const vec_value_t>& weights
) const
{
    base_t::check_cmul(j, v.size(), weights.size(), rows(), cols());
    vec_value_t buff(_n_threads * (_n_threads > 1) * !util::omp_in_parallel());
    return _cmul(j, v, weights, buff);
}

ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_DENSE::ctmul(
    int j, 
    value_t v, 
    Eigen::Ref<vec_value_t> out
) 
{
    base_t::check_ctmul(j, out.size(), rows(), cols());
    _ctmul(j, v, out, _n_threads);
}

ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_DENSE::bmul(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) 
{
    base_t::check_bmul(j, q, v.size(), weights.size(), out.size(), rows(), cols());
    _bmul(j, q, v, weights, out, _buff);
}

ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_DENSE::bmul_safe(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
{
    base_t::check_bmul(j, q, v.size(), weights.size(), out.size(), rows(), cols());
    vec_value_t buffer(_n_threads * (_n_threads > 1) * !util::omp_in_parallel() * _mat.cols());
    _bmul(j, q, v, weights, out, buffer);
}

ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_DENSE::btmul(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    Eigen::Ref<vec_value_t> out
) 
{
    base_t::check_btmul(j, q, v.size(), out.size(), rows(), cols());
    const auto n = _mat.rows();
    const auto d = _mat.cols();
    int n_processed = 0;
    while (n_processed < q) {
        auto k = j + n_processed;
        const auto k_m = k / d;
        k -= k_m * d;
        const auto k_d = k;
        const auto size = std::min<int>(d-k_d, q-n_processed);
        Eigen::Map<vec_value_t> Xv(_buff.data(), n);
        Eigen::Map<rowmat_value_t> buff(_buff.data() + n, _n_threads, n);
        auto Xv_m = Xv.matrix();
        dgemv(
            _mat.middleCols(k_d, size).transpose(),
            v.segment(n_processed, size).matrix(),
            _n_threads,
            buff,
            Xv_m
        );
        dvaddi(
            out, 
            Xv * _mask.col(k_m).transpose().template cast<value_t>().array(), 
            _n_threads
        );
        n_processed += size;
    }
}

ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_DENSE::mul(
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const 
{
    const auto d = _mat.cols();
    const auto m = _mask.cols();
    // NOTE: MSVC does not like it when we try to capture v_weights and buff.
    // This is a bug in MSVC (god I hate microsoft so much...).
    const auto routine = [&](auto i, const auto& v_weights) {
        Eigen::Map<rowmat_value_t> buff(out.data(), _n_threads, d);
        auto out_m = out.segment(i * d, d).matrix();
        dgemv(
            _mat,
            _mask.col(i).transpose().template cast<value_t>().cwiseProduct(v_weights),
            1,
            buff /* unused */,
            out_m
        );
    };
    const auto v_weights = (v * weights).matrix();
    util::omp_parallel_for([&](auto i) { routine(i, v_weights); }, 0, m, _n_threads);
}

ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_DENSE_TP
int
ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_DENSE::rows() const
{
    return _mat.rows();
}

ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_DENSE_TP
int
ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_DENSE::cols() const
{
    return _mat.cols() * _mask.cols();
}

ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_DENSE::cov(
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

    const auto n = _mat.rows();
    const auto d = _mat.cols();
    colmat_value_t buffer(n, q);

    int n_processed = 0;
    while (n_processed < q) {
        auto k = j + n_processed;
        const auto k_m = k / d;
        k -= k_m * d;
        const auto k_d = k;
        const auto size = std::min<int>(d-k_d, q-n_processed);
        const auto mat = _mat.middleCols(k_d, size);
        const auto mask = _mask.col(k_m);

        auto curr_block = buffer.middleCols(n_processed, size).array();
        curr_block.array() = (
            mat.array().colwise() * 
            mask.template cast<value_t>().cwiseProduct(sqrt_weights.matrix().transpose()).array()
        );
        n_processed += size;
    }

    vec_value_t outs(q * q * _n_threads);
    dxtx(buffer, _n_threads, outs, out);
}

ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_DENSE::sq_mul(
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
{
    const auto d = _mat.cols();
    const auto m = _mask.cols();
    colmat_value_t mat_sq = _mat.array().square().matrix();
    // NOTE: MSVC does not like it when we try to capture v_weights and buff.
    // This is a bug in MSVC (god I hate microsoft so much...).
    const auto routine = [&](auto i, const auto& w) {
        Eigen::Map<rowmat_value_t> buff(out.data(), _n_threads, d);
        auto out_m = out.segment(i * d, d).matrix();
        dgemv(
            mat_sq,
            _mask.col(i).transpose().template cast<value_t>().cwiseProduct(w.matrix()),
            1,
            buff /* unused */,
            out_m
        );
    };
    util::omp_parallel_for([&](auto i) { routine(i, weights); }, 0, m, _n_threads);
}

ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_DENSE::sp_tmul(
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

ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_SPARSE_TP
typename ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_SPARSE::value_t 
ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_SPARSE::_cmul(
    int j, 
    const Eigen::Ref<const vec_value_t>& v,
    const Eigen::Ref<const vec_value_t>& weights,
    size_t n_threads,
    Eigen::Ref<vec_value_t> buff
) const
{
    const auto d = _mat.cols();
    const auto j_m = j / d;
    j -= j_m * d;
    const auto j_d = j;
    const auto outer = _mat.outerIndexPtr();
    const auto outer_j_d = outer[j_d];
    const auto size_j_d = outer[j_d+1] - outer_j_d;
    const Eigen::Map<const vec_sp_index_t> inner_j_d(
        _mat.innerIndexPtr() + outer_j_d,
        size_j_d
    );
    const Eigen::Map<const vec_sp_value_t> value_j_d(
        _mat.valuePtr() + outer_j_d,
        size_j_d
    );
    return spddot(
        inner_j_d,
        value_j_d,
        (v * weights * _mask.col(j_m).transpose().array().template cast<value_t>()),
        n_threads,
        buff
    );
}

ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_SPARSE_TP
void 
ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_SPARSE::_ctmul(
    int j, 
    value_t v, 
    Eigen::Ref<vec_value_t> out,
    size_t n_threads
) const
{
    const auto d = _mat.cols();
    const auto j_m = j / d;
    j -= j_m * d;
    const auto j_d = j;
    const auto outer = _mat.outerIndexPtr();
    const auto outer_j_d = outer[j_d];
    const auto size_j_d = outer[j_d+1] - outer_j_d;
    const Eigen::Map<const vec_sp_index_t> inner_j_d(
        _mat.innerIndexPtr() + outer_j_d,
        size_j_d
    );
    const Eigen::Map<const vec_sp_value_t> value_j_d(
        _mat.valuePtr() + outer_j_d,
        size_j_d
    );
    spdaddi(
        inner_j_d, 
        value_j_d, 
        v * _mask.col(j_m).transpose().array().template cast<value_t>(),
        out, 
        n_threads
    );
}

ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_SPARSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_SPARSE::_bmul(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out,
    Eigen::Ref<vec_value_t> buff
) const
{
    for (int k = 0; k < q; ++k) {
        out[k] = _cmul(j+k, v, weights, _n_threads, buff);
    }
}

ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_SPARSE_TP
ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_SPARSE::MatrixNaiveConvexGatedReluSparse(
    size_t rows,
    size_t cols,
    size_t nnz,
    const Eigen::Ref<const vec_sp_index_t>& outer,
    const Eigen::Ref<const vec_sp_index_t>& inner,
    const Eigen::Ref<const vec_sp_value_t>& value,
    const Eigen::Ref<const mask_t>& mask,
    size_t n_threads
):
    _mat(rows, cols, nnz, outer.data(), inner.data(), value.data()),
    _mask(mask.data(), mask.rows(), mask.cols()),
    _n_threads(n_threads),
    _buff(n_threads)
{
    const Eigen::Index n = rows;

    if (mask.rows() != n) {
        throw util::adelie_core_error("mask must be (n, m) where mat is (n, d).");
    }
    if (n_threads < 1) {
        throw util::adelie_core_error("n_threads must be >= 1.");
    }
}

ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_SPARSE_TP
typename ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_SPARSE::value_t
ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_SPARSE::cmul(
    int j, 
    const Eigen::Ref<const vec_value_t>& v,
    const Eigen::Ref<const vec_value_t>& weights
) 
{
    base_t::check_cmul(j, v.size(), weights.size(), rows(), cols());
    return _cmul(j, v, weights, _n_threads, _buff);
}

ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_SPARSE_TP
typename ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_SPARSE::value_t
ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_SPARSE::cmul_safe(
    int j, 
    const Eigen::Ref<const vec_value_t>& v,
    const Eigen::Ref<const vec_value_t>& weights
) const
{
    base_t::check_cmul(j, v.size(), weights.size(), rows(), cols());
    vec_value_t buff(_n_threads * (_n_threads > 1) * !util::omp_in_parallel());
    return _cmul(j, v, weights, _n_threads, buff);
}

ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_SPARSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_SPARSE::ctmul(
    int j, 
    value_t v, 
    Eigen::Ref<vec_value_t> out
) 
{
    base_t::check_ctmul(j, out.size(), rows(), cols());
    _ctmul(j, v, out, _n_threads);
}

ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_SPARSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_SPARSE::bmul(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) 
{
    base_t::check_bmul(j, q, v.size(), weights.size(), out.size(), rows(), cols());
    _bmul(j, q, v, weights, out, _buff);
}

ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_SPARSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_SPARSE::bmul_safe(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
{
    base_t::check_bmul(j, q, v.size(), weights.size(), out.size(), rows(), cols());
    vec_value_t buff(_n_threads * (_n_threads > 1) * !util::omp_in_parallel());
    _bmul(j, q, v, weights, out, buff);
}

ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_SPARSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_SPARSE::btmul(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    Eigen::Ref<vec_value_t> out
) 
{
    base_t::check_btmul(j, q, v.size(), out.size(), rows(), cols());
    for (int k = 0; k < q; ++k) {
        _ctmul(j+k, v[k], out, _n_threads);
    }
}

ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_SPARSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_SPARSE::mul(
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
{
    const auto d = _mat.cols();
    const auto m = _mask.cols();
    const auto routine = [&](int k) {
        out[k] = _cmul(k, v, weights, 1, out /* unused */);
    };
    util::omp_parallel_for(routine, 0, m*d, _n_threads);
}

ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_SPARSE_TP
int
ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_SPARSE::rows() const
{
    return _mat.rows();
}

ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_SPARSE_TP
int
ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_SPARSE::cols() const
{
    return _mat.cols() * _mask.cols();
}

ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_SPARSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_SPARSE::cov(
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

    const auto d = _mat.cols();

    const auto routine = [&](int i1) {
        auto index_1 = j+i1;
        const auto index_1_m = index_1 / d;
        index_1 -= index_1_m * d;
        const auto outer_1 = _mat.outerIndexPtr()[index_1];
        const auto size_1 = _mat.outerIndexPtr()[index_1+1] - outer_1;
        const Eigen::Map<const vec_sp_index_t> inner_1(
            _mat.innerIndexPtr() + outer_1, size_1
        );
        const Eigen::Map<const vec_sp_value_t> value_1(
            _mat.valuePtr() + outer_1, size_1
        );
        const auto mask_1 = _mask.col(index_1_m).transpose().array().template cast<value_t>();
        for (int i2 = 0; i2 <= i1; ++i2) {
            auto index_2 = j+i2;
            const auto index_2_m = index_2 / d;
            index_2 -= index_2_m * d;
            const auto outer_2 = _mat.outerIndexPtr()[index_2];
            const auto size_2 = _mat.outerIndexPtr()[index_2+1] - outer_2;
            const Eigen::Map<const vec_sp_index_t> inner_2(
                _mat.innerIndexPtr() + outer_2, size_2
            );
            const Eigen::Map<const vec_sp_value_t> value_2(
                _mat.valuePtr() + outer_2, size_2
            );
            const auto mask_2 = _mask.col(index_2_m).transpose().array().template cast<value_t>();

            out(i1, i2) = svsvwdot(
                inner_1, value_1,
                inner_2, value_2,
                sqrt_weights.square() * mask_1 * mask_2
            );
        }
    };
    util::omp_parallel_for(routine, 0, q, _n_threads);
    for (int i1 = 0; i1 < q; ++i1) {
        for (int i2 = i1+1; i2 < q; ++i2) {
            out(i1, i2) = out(i2, i1);
        }
    }
}

ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_SPARSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_SPARSE::sq_mul(
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
{
    const auto d = _mat.cols();
    const auto m = _mask.cols();
    Eigen::SparseMatrix<value_t, Eigen::ColMajor> mat_sq = _mat.cwiseProduct(_mat);
    // NOTE: MSVC does not like it when we try to capture mat_sq.
    // This is a bug in MSVC (god I hate microsoft so much...).
    const auto routine = [&](int k, const auto& mat_sq) {
        out.segment(k * d, d).matrix() = (
            (weights * _mask.col(k).transpose().array().template cast<value_t>()).matrix()
        ) * mat_sq;
    };
    util::omp_parallel_for([&](auto k) { routine(k, mat_sq); }, 0, m, _n_threads);
}

ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_SPARSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_SPARSE::sp_tmul(
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