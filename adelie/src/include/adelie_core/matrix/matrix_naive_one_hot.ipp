#pragma once
#include <adelie_core/matrix/matrix_naive_one_hot.hpp>
#include <adelie_core/matrix/utils.hpp>

namespace adelie_core {
namespace matrix {

ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE_TP
auto
ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE::init_outer(
    const Eigen::Ref<const vec_index_t>& levels
)
{
    vec_index_t outer(levels.size() + 1);
    outer[0] = 0;
    for (int i = 0; i < levels.size(); ++i) {
        const auto li = levels[i];
        const auto size = std::max<size_t>(li, 1);
        outer[i+1] = outer[i] + size; 
    }
    return outer;
}

ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE_TP
auto
ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE::init_slice_map(
    const Eigen::Ref<const vec_index_t>& levels,
    size_t cols
)
{
    vec_index_t slice_map(cols);
    size_t begin = 0;
    for (int i = 0; i < levels.size(); ++i) {
        const auto li = levels[i];
        const auto block_size = std::max<size_t>(li, 1);
        for (size_t j = 0; j < block_size; ++j) {
            slice_map[begin + j] = i;
        }
        begin += block_size;
    }
    return slice_map;
}

ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE_TP
auto
ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE::init_index_map(
    const Eigen::Ref<const vec_index_t>& levels,
    size_t cols
)
{
    vec_index_t index_map(cols);
    size_t begin = 0;
    for (int i = 0; i < levels.size(); ++i) {
        const auto li = levels[i];
        const auto block_size = std::max<size_t>(li, 1);
        for (size_t j = 0; j < block_size; ++j) {
            index_map[begin + j] = j;
        }
        begin += block_size;
    }
    return index_map;
}

ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE_TP
typename ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE::value_t
ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE::_cmul(
    int j, 
    const Eigen::Ref<const vec_value_t>& v,
    const Eigen::Ref<const vec_value_t>& weights,
    size_t n_threads,
    Eigen::Ref<vec_value_t> buff
) const
{
    const auto& w = weights;
    const auto slice = _slice_map[j];
    const auto index = _index_map[j];
    const auto level = std::max<size_t>(_levels[slice], 0);

    switch (level) {
        case 0: {
            return ddot((v * w).matrix(), _mat.col(slice), n_threads, buff);
            break;
        }
        case 1: {
            return ddot(v.matrix(), w.matrix(), n_threads, buff);
            break;
        }
        default: {
            const auto m_slice = _mat.col(slice).transpose().array();
            return ddot(
                (v * w).matrix(),
                (m_slice == index).template cast<value_t>().matrix(),
                n_threads,
                buff
            );
            break;
        }
    }
} 

ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE_TP
typename ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE::value_t
ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE::_sq_cmul(
    int j, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> buff
) const
{
    constexpr size_t n_threads = 1;
    const auto& w = weights;
    const auto slice = _slice_map[j];
    const auto index = _index_map[j];
    const auto level = std::max<size_t>(_levels[slice], 0);

    switch (level) {
        case 0: {
            return ddot(w.matrix(), _mat.col(slice).array().square().matrix(), n_threads, buff);
            break;
        }
        case 1: {
            return w.sum();
            break;
        }
        default: {
            const auto m_slice = _mat.col(slice).transpose().array();
            return ddot(
                w.matrix(),
                (m_slice == index).template cast<value_t>().matrix(),
                n_threads,
                buff
            );
            break;
        }
    }
} 

ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE::_ctmul(
    int j, 
    value_t v, 
    Eigen::Ref<vec_value_t> out,
    size_t n_threads
) const
{
    const auto slice = _slice_map[j];
    const auto index = _index_map[j];
    const auto level = std::max<size_t>(_levels[slice], 0);

    switch (level) {
        case 0: {
            dvaddi(out, v * _mat.col(slice).transpose().array(), n_threads);
            break;
        }
        case 1: {
            dvaddi(
                out, 
                vec_value_t::NullaryExpr(_mat.rows(), [=](auto) { 
                    return v; 
                }), 
                n_threads
            );
            break;
        }
        default: {
            const auto m_slice = _mat.col(slice).transpose().array();
            dvaddi(
                out,
                v * (m_slice == index).template cast<value_t>(),
                n_threads
            );
            break;
        }
    }
}

ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE::_bmul(
    int begin,
    int slice,
    int index,
    int level,
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out,
    Eigen::Ref<vec_value_t> buff,
    size_t n_threads
) const
{
    const size_t size = out.size();
    const size_t full_size = std::max<size_t>(level, 1);
    if (index != 0 || size != full_size) {
        for (size_t l = 0; l < size; ++l) {
            out[l] = _cmul(begin+l, v, weights, n_threads, buff);
        }
        return;
    }
    const auto& w = weights;
    level = std::max<size_t>(level, 0);
    switch (level) {
        case 0: 
        case 1: {
            out[0] = _cmul(begin, v, weights, n_threads, buff);
            break;
        }
        default: {
            out.setZero();
            for (int i = 0; i < _mat.rows(); ++i) {
                const auto val = v[i] * w[i];
                const int k = _mat(i, slice);
                out[k] += val;
            }
            break;
        }
    }
}

ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE::_sq_bmul(
    int begin,
    int slice,
    int level,
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out,
    Eigen::Ref<vec_value_t> buff
) const
{
    const auto& w = weights;
    level = std::max<size_t>(level, 0);
    switch (level) {
        case 0: 
        case 1: {
            out[0] = _sq_cmul(begin, weights, buff);
            break;
        }
        default: {
            out.setZero();
            for (int i = 0; i < _mat.rows(); ++i) {
                const auto val = w[i];
                const int k = _mat(i, slice);
                out[k] += val;
            }
            break;
        }
    }
}

ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE::_btmul(
    int begin,
    int slice,
    int index,
    int level,
    size_t size,
    const Eigen::Ref<const vec_value_t>& v, 
    Eigen::Ref<vec_value_t> out,
    size_t n_threads
) const
{
    const auto full_size = std::max<size_t>(level, 1);
    if (index != 0 || size != full_size) {
        for (size_t l = 0; l < size; ++l) {
            _ctmul(begin+l, v[l], out, n_threads);
        }
        return;
    }
    level = std::max(level, 0);
    switch (level) {
        case 0: {
            dvaddi(out, v[0] * _mat.col(slice).transpose().array(), n_threads);
            break;
        }
        case 1: {
            dvaddi(
                out, 
                vec_value_t::NullaryExpr(_mat.rows(), [&](auto) { 
                    return v[0];
                }), 
                n_threads
            );
            break;
        }
        default: {
            dvaddi(
                out, 
                vec_value_t::NullaryExpr(_mat.rows(), [&](auto i) {
                    const int k = _mat(i, slice);
                    return v[k];
                }),
                n_threads
            );
            break;
        }
    }
}

ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE_TP
ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE::MatrixNaiveOneHotDense(
    const Eigen::Ref<const dense_t>& mat,
    const Eigen::Ref<const vec_index_t>& levels,
    size_t n_threads
):
    _mat(mat.data(), mat.rows(), mat.cols()),
    _levels(levels.data(), levels.size()),
    _outer(init_outer(levels)),
    _cols(_outer[_outer.size()-1]),
    _slice_map(init_slice_map(levels, _cols)),
    _index_map(init_index_map(levels, _cols)),
    _n_threads(n_threads),
    _buff(n_threads)
{
    const auto d = mat.cols();

    if (levels.size() != d) {
        throw util::adelie_core_error("levels must be (d,) where mat is (n, d).");
    }
    if (n_threads < 1) {
        throw util::adelie_core_error("n_threads must be >= 1.");
    }
}

ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE_TP
typename ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE::value_t
ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE::cmul(
    int j, 
    const Eigen::Ref<const vec_value_t>& v,
    const Eigen::Ref<const vec_value_t>& weights
)
{
    base_t::check_cmul(j, v.size(), weights.size(), rows(), cols());
    return _cmul(j, v, weights, _n_threads, _buff);
}

ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE_TP
typename ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE::value_t
ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE::cmul_safe(
    int j, 
    const Eigen::Ref<const vec_value_t>& v,
    const Eigen::Ref<const vec_value_t>& weights
) const
{
    base_t::check_cmul(j, v.size(), weights.size(), rows(), cols());
    vec_value_t buff(_n_threads * (_n_threads > 1) * !util::omp_in_parallel());
    return _cmul(j, v, weights, _n_threads, buff);
}

ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE::ctmul(
    int j, 
    value_t v, 
    Eigen::Ref<vec_value_t> out
)
{
    base_t::check_ctmul(j, out.size(), rows(), cols());
    _ctmul(j, v, out, _n_threads);
}

ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE::bmul(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
)
{
    base_t::check_bmul(j, q, v.size(), weights.size(), out.size(), rows(), cols());
    int n_processed = 0;
    while (n_processed < q) {
        const auto jj = j + n_processed;
        const auto slice = _slice_map[jj];
        const auto index = _index_map[jj];
        const auto level = _levels[slice];
        const auto full_size = std::max<size_t>(level, 1);
        const auto size = std::min<size_t>(full_size - index, q - n_processed);
        auto out_curr = out.segment(n_processed, size);
        _bmul(jj, slice, index, level, v, weights, out_curr, _buff, _n_threads);
        n_processed += size;
    }
}

ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE::bmul_safe(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
{
    base_t::check_bmul(j, q, v.size(), weights.size(), out.size(), rows(), cols());
    vec_value_t buff(_n_threads * (_n_threads > 1) * !util::omp_in_parallel());
    int n_processed = 0;
    while (n_processed < q) {
        const auto jj = j + n_processed;
        const auto slice = _slice_map[jj];
        const auto index = _index_map[jj];
        const auto level = _levels[slice];
        const auto full_size = std::max<size_t>(level, 1);
        const auto size = std::min<size_t>(full_size - index, q - n_processed);
        auto out_curr = out.segment(n_processed, size);
        _bmul(jj, slice, index, level, v, weights, out_curr, buff, _n_threads);
        n_processed += size;
    }
}

ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE::btmul(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    Eigen::Ref<vec_value_t> out
)
{
    base_t::check_btmul(j, q, v.size(), out.size(), rows(), cols());
    int n_processed = 0;
    while (n_processed < q) {
        const auto jj = j + n_processed;
        const auto slice = _slice_map[jj];
        const auto index = _index_map[jj];
        const auto level = _levels[slice];
        const auto full_size = std::max<size_t>(level, 1);
        const auto size = std::min<size_t>(full_size - index, q - n_processed);
        const auto v_curr = v.segment(n_processed, size);
        _btmul(jj, slice, index, level, size, v_curr, out, _n_threads);
        n_processed += size;
    }
}

ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE::mul(
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
{
    const auto routine = [&](auto g) {
        const auto j = _outer[g];
        const auto level = _levels[g];
        const auto full_size = std::max<size_t>(level, 1);
        auto out_curr = out.segment(j, full_size);
        _bmul(j, g, 0, level, v, weights, out_curr, out /* unused */, 1);
    };
    util::omp_parallel_for(routine, 0, _mat.cols(), _n_threads);
}

ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE_TP
int
ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE::rows() const
{
    return _mat.rows();
}
    
ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE_TP
int
ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE::cols() const
{
    return _cols;
}

ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE::cov(
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

    const auto slice = _slice_map[j];
    const auto index = _index_map[j];
    const auto outer = _outer[slice];
    if ((index != 0) || (_outer[slice+1] - outer != q)) {
        throw util::adelie_core_error(
            "MatrixNaiveOneHotDense::cov() not implemented for ranges that contain multiple blocks. "
            "If triggered from a solver, this error is usually because "
            "the groups argument is inconsistent with the implicit group structure "
            "of the matrix. "
        );
    }

    const auto& sqrt_w = sqrt_weights;
    const auto level = std::max<size_t>(_levels[slice], 0);

    vec_value_t buff(_n_threads * (_n_threads > 1) * !util::omp_in_parallel());

    switch (level) {
        case 0: {
            auto mi = _mat.col(slice).transpose().array();
            const auto sqrt_w_mi = (sqrt_w * mi).matrix();
            out(0, 0) = ddot(sqrt_w_mi, sqrt_w_mi, _n_threads, buff);
            break;
        }
        case 1: {
            out(0, 0) = ddot(sqrt_w.matrix(), sqrt_w.matrix(), _n_threads, buff);
            break;
        }
        default: {
            out.setZero();
            for (int i = 0; i < _mat.rows(); ++i) {
                const auto sqrt_wi = sqrt_w[i];
                const int k = _mat(i, slice);
                out(k, k) += sqrt_wi * sqrt_wi;
            }
            break;
        }
    }
}

ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE::sq_mul(
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
{
    const auto routine = [&](auto g) {
        const auto j = _outer[g];
        const auto level = _levels[g];
        const auto full_size = std::max<size_t>(level, 1);
        auto out_curr = out.segment(j, full_size);
        _sq_bmul(j, g, level, weights, out_curr, out /* unused */);
    };
    util::omp_parallel_for(routine, 0, _mat.cols(), _n_threads);
}

ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE::sp_tmul(
    const sp_mat_value_t& v, 
    Eigen::Ref<rowmat_value_t> out
) const
{
    base_t::check_sp_tmul(
        v.rows(), v.cols(), out.rows(), out.cols(), rows(), cols()
    );
    const auto routine = [&](auto k) {
        typename sp_mat_value_t::InnerIterator it(v, k);
        auto out_k = out.row(k);
        out_k.setZero();
        for (; it; ++it) {
            _ctmul(it.index(), it.value(), out_k, 1);
        }
    };
    util::omp_parallel_for(routine, 0, v.outerSize(), _n_threads);
}

ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE::mean(
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const 
{
    base_t::mean(weights, out);
    const auto d = _mat.cols();
    for (int g = 0; g < d; ++g) {
        const auto j = _outer[g];
        const auto level = _levels[g];
        if (level <= 0) continue;
        out.segment(j, level) = 0;
    }
}

ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_ONE_HOT_DENSE::var(
    const Eigen::Ref<const vec_value_t>& centers,
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
{
    base_t::var(centers, weights, out);
    const auto d = _mat.cols();
    for (int g = 0; g < d; ++g) {
        const auto j = _outer[g];
        const auto level = _levels[g];
        if (level <= 0) continue;
        out.segment(j, level) = 1;
    }
}

} // namespace matrix
} // namespace adelie_core 