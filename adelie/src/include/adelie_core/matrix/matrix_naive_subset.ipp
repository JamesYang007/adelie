#pragma once
#include <adelie_core/matrix/matrix_naive_subset.hpp>
#include <adelie_core/util/omp.hpp>

namespace adelie_core {
namespace matrix {

ADELIE_CORE_MATRIX_NAIVE_CSUBSET_TP
auto
ADELIE_CORE_MATRIX_NAIVE_CSUBSET::init_subset_cinfo(
    const Eigen::Ref<const vec_index_t>& subset
)
{
    if (subset.size() == 0) {
        throw util::adelie_core_error(
            "subset must be non-empty."
        );
    }

    vec_index_t subset_csize(subset.size());
    dyn_vec_index_t subset_cbegin;
    subset_cbegin.reserve(subset.size());

    size_t count = 1;
    size_t begin = 0;
    for (size_t i = 1; i < static_cast<size_t>(subset.size()); ++i) {
        if (subset[i] == subset[i-1] + 1) {
            ++count;
            continue;
        }
        for (size_t j = 0; j < count; ++j) {
            subset_csize[begin+j] = count - j;
        }
        subset_cbegin.push_back(begin);
        begin += count;
        count = 1;
    }
    if (begin != static_cast<size_t>(subset.size())) {
        for (size_t j = 0; j < count; ++j) {
            subset_csize[begin+j] = count - j;
        }
        subset_cbegin.push_back(begin);
    }
    return std::make_tuple(subset_csize, subset_cbegin);
}

ADELIE_CORE_MATRIX_NAIVE_CSUBSET_TP
ADELIE_CORE_MATRIX_NAIVE_CSUBSET::MatrixNaiveCSubset(
    base_t& mat,
    const Eigen::Ref<const vec_index_t>& subset,
    size_t n_threads
): 
    _mat(&mat),
    _subset(subset.data(), subset.size()),
    _subset_cinfo(init_subset_cinfo(subset)),
    _n_threads(n_threads)
{
    if ((subset.minCoeff() < 0) || (subset.maxCoeff() >= mat.cols())) {
        throw util::adelie_core_error(
            "subset must contain unique values in the range [0, p) "
            "where mat is (n, p)."
        ) ;
    }
    if (n_threads < 1) {
        throw util::adelie_core_error("n_threads must be >= 1.");
    }
}

ADELIE_CORE_MATRIX_NAIVE_CSUBSET_TP
typename ADELIE_CORE_MATRIX_NAIVE_CSUBSET::value_t
ADELIE_CORE_MATRIX_NAIVE_CSUBSET::cmul(
    int j, 
    const Eigen::Ref<const vec_value_t>& v,
    const Eigen::Ref<const vec_value_t>& weights
)
{
    base_t::check_cmul(j, v.size(), weights.size(), rows(), cols());
    return _mat->cmul(_subset[j], v, weights);
}

ADELIE_CORE_MATRIX_NAIVE_CSUBSET_TP
typename ADELIE_CORE_MATRIX_NAIVE_CSUBSET::value_t
ADELIE_CORE_MATRIX_NAIVE_CSUBSET::cmul_safe(
    int j, 
    const Eigen::Ref<const vec_value_t>& v,
    const Eigen::Ref<const vec_value_t>& weights
) const
{
    base_t::check_cmul(j, v.size(), weights.size(), rows(), cols());
    return _mat->cmul_safe(_subset[j], v, weights);
}

ADELIE_CORE_MATRIX_NAIVE_CSUBSET_TP
void
ADELIE_CORE_MATRIX_NAIVE_CSUBSET::ctmul(
    int j, 
    value_t v, 
    Eigen::Ref<vec_value_t> out
)
{
    base_t::check_ctmul(j, out.size(), rows(), cols());
    _mat->ctmul(_subset[j], v, out);
}

ADELIE_CORE_MATRIX_NAIVE_CSUBSET_TP
void
ADELIE_CORE_MATRIX_NAIVE_CSUBSET::bmul(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
)
{
    base_t::check_bmul(j, q, v.size(), weights.size(), out.size(), rows(), cols());
    const auto& _subset_csize = std::get<0>(_subset_cinfo);
    int n_processed = 0;
    while (n_processed < q) {
        const auto k = j + n_processed;
        const auto size = std::min<size_t>(_subset_csize[k], q-n_processed);
        if (size == 1) {
            out[n_processed] = _mat->cmul(_subset[k], v, weights);
        } else {
            auto curr_out = out.segment(n_processed, size);
            _mat->bmul(_subset[k], size, v, weights, curr_out);
        }
        n_processed += size;
    }
}

ADELIE_CORE_MATRIX_NAIVE_CSUBSET_TP
void
ADELIE_CORE_MATRIX_NAIVE_CSUBSET::bmul_safe(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
{
    base_t::check_bmul(j, q, v.size(), weights.size(), out.size(), rows(), cols());
    const auto& _subset_csize = std::get<0>(_subset_cinfo);
    int n_processed = 0;
    while (n_processed < q) {
        const auto k = j + n_processed;
        const auto size = std::min<size_t>(_subset_csize[k], q-n_processed);
        if (size == 1) {
            out[n_processed] = _mat->cmul_safe(_subset[k], v, weights);
        } else {
            auto curr_out = out.segment(n_processed, size);
            _mat->bmul_safe(_subset[k], size, v, weights, curr_out);
        }
        n_processed += size;
    }
}

ADELIE_CORE_MATRIX_NAIVE_CSUBSET_TP
void
ADELIE_CORE_MATRIX_NAIVE_CSUBSET::btmul(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    Eigen::Ref<vec_value_t> out
)
{
    base_t::check_btmul(j, q, v.size(), out.size(), rows(), cols());
    const auto& _subset_csize = std::get<0>(_subset_cinfo);
    int n_processed = 0;
    while (n_processed < q) {
        const auto k = j + n_processed;
        const auto size = std::min<size_t>(_subset_csize[k], q-n_processed);
        if (size == 1) {
            _mat->ctmul(_subset[k], v[n_processed], out);
        } else {
            const auto curr_v = v.segment(n_processed, size);
            _mat->btmul(_subset[k], size, curr_v, out);
        }
        n_processed += size;
    }
}

ADELIE_CORE_MATRIX_NAIVE_CSUBSET_TP
void
ADELIE_CORE_MATRIX_NAIVE_CSUBSET::mul(
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
{
    const auto& _subset_csize = std::get<0>(_subset_cinfo);
    const auto& _subset_cbegin = std::get<1>(_subset_cinfo);

    const auto routine = [&](auto t) {
        const auto subset_idx = _subset_cbegin[t];
        const auto j = _subset[subset_idx];
        const auto q = _subset_csize[subset_idx];
        auto curr_out = out.segment(subset_idx, q);
        _mat->bmul_safe(j, q, v, weights, curr_out);
    };
    util::omp_parallel_for(routine, 0, _subset_cbegin.size(), _n_threads * (_n_threads <= _subset_cbegin.size()));
}

ADELIE_CORE_MATRIX_NAIVE_CSUBSET_TP
int
ADELIE_CORE_MATRIX_NAIVE_CSUBSET::rows() const
{
    return _mat->rows();
}

ADELIE_CORE_MATRIX_NAIVE_CSUBSET_TP
int
ADELIE_CORE_MATRIX_NAIVE_CSUBSET::cols() const
{
    return _subset.size();
}

ADELIE_CORE_MATRIX_NAIVE_CSUBSET_TP
void
ADELIE_CORE_MATRIX_NAIVE_CSUBSET::cov(
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
    const auto& _subset_csize = std::get<0>(_subset_cinfo);
    if (_subset_csize[j] < q) {
        throw util::adelie_core_error(
            "MatrixNaiveCSubset::cov() is not implemented when "
            "subset[j:j+q] is not contiguous. "
        );
    }
    _mat->cov(_subset[j], q, sqrt_weights, out);
}

ADELIE_CORE_MATRIX_NAIVE_CSUBSET_TP
void
ADELIE_CORE_MATRIX_NAIVE_CSUBSET::sq_mul(
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
{
    const auto& _subset_csize = std::get<0>(_subset_cinfo);
    const auto& _subset_cbegin = std::get<1>(_subset_cinfo);

    vec_value_t sq_means(_mat->cols());
    _mat->sq_mul(weights, sq_means);
    for (int t = 0; t < static_cast<int>(_subset_cbegin.size()); ++t) {
        const auto subset_idx = _subset_cbegin[t];
        const auto j = _subset[subset_idx];
        const auto q = _subset_csize[subset_idx];
        auto curr_out = out.segment(subset_idx, q);
        curr_out = sq_means.segment(j, q);
    }
}

ADELIE_CORE_MATRIX_NAIVE_CSUBSET_TP
void
ADELIE_CORE_MATRIX_NAIVE_CSUBSET::sp_tmul(
    const sp_mat_value_t& v, 
    Eigen::Ref<rowmat_value_t> out
) const
{
    base_t::check_sp_tmul(
        v.rows(), v.cols(), out.rows(), out.cols(), rows(), cols()
    );

    for (int k = 0; k < v.outerSize(); ++k) {
        typename sp_mat_value_t::InnerIterator it(v, k);
        auto out_k = out.row(k);
        out_k.setZero();
        for (; it; ++it) {
            _mat->ctmul(_subset[it.index()], it.value(), out_k);
        }
    }
}

ADELIE_CORE_MATRIX_NAIVE_CSUBSET_TP
void
ADELIE_CORE_MATRIX_NAIVE_CSUBSET::mean(
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
{
    vec_value_t m(_mat->cols());
    _mat->mean(weights, m);
    for (int i = 0; i < _subset.size(); ++i) {
        out[i] = m[_subset[i]];
    }
}

ADELIE_CORE_MATRIX_NAIVE_CSUBSET_TP
void
ADELIE_CORE_MATRIX_NAIVE_CSUBSET::var(
    const Eigen::Ref<const vec_value_t>& centers,
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
{
    vec_value_t new_c(_mat->cols());
    new_c.setZero();
    for (int i = 0; i < _subset.size(); ++i) {
        new_c[_subset[i]] = centers[i];
    }

    vec_value_t v(_mat->cols());
    _mat->var(new_c, weights, v);
    for (int i = 0; i < _subset.size(); ++i) {
        out[i] = v[_subset[i]];
    }
}

ADELIE_CORE_MATRIX_NAIVE_RSUBSET_TP
auto
ADELIE_CORE_MATRIX_NAIVE_RSUBSET::init_mask(
    size_t n,
    const Eigen::Ref<const vec_index_t>& subset
)
{
    if (subset.size() == 0) {
        throw util::adelie_core_error(
            "subset must be non-empty."
        );
    }

    vec_value_t mask(n);
    mask.setZero();
    for (int i = 0; i < subset.size(); ++i) {
        mask[subset[i]] = true;
    } 
    return mask;
}

ADELIE_CORE_MATRIX_NAIVE_RSUBSET_TP
ADELIE_CORE_MATRIX_NAIVE_RSUBSET::MatrixNaiveRSubset(
    base_t& mat,
    const Eigen::Ref<const vec_index_t>& subset,
    size_t n_threads
): 
    _mat(&mat),
    _subset(subset.data(), subset.size()),
    _mask(init_mask(mat.rows(), subset)),
    _n_threads(n_threads),
    _buffer(mat.rows())
{
    if ((subset.minCoeff() < 0) || (subset.maxCoeff() >= mat.rows())) {
        throw util::adelie_core_error(
            "subset must contain unique values in the range [0, n) "
            "where mat is (n, p)."
        ) ;
    }
    if (n_threads < 1) {
        throw util::adelie_core_error("n_threads must be >= 1.");
    }
}

ADELIE_CORE_MATRIX_NAIVE_RSUBSET_TP
typename ADELIE_CORE_MATRIX_NAIVE_RSUBSET::value_t
ADELIE_CORE_MATRIX_NAIVE_RSUBSET::cmul(
    int j, 
    const Eigen::Ref<const vec_value_t>& v,
    const Eigen::Ref<const vec_value_t>& weights
)
{
    base_t::check_cmul(j, v.size(), weights.size(), rows(), cols());
    _buffer.setZero();
    for (int i = 0; i < _subset.size(); ++i) {
        _buffer[_subset[i]] = v[i] * weights[i];
    }
    return _mat->cmul(j, _mask, _buffer);
}

ADELIE_CORE_MATRIX_NAIVE_RSUBSET_TP
typename ADELIE_CORE_MATRIX_NAIVE_RSUBSET::value_t
ADELIE_CORE_MATRIX_NAIVE_RSUBSET::cmul_safe(
    int j, 
    const Eigen::Ref<const vec_value_t>& v,
    const Eigen::Ref<const vec_value_t>& weights
) const
{
    base_t::check_cmul(j, v.size(), weights.size(), rows(), cols());
    vec_value_t buffer(_mat->rows());
    buffer.setZero();
    for (int i = 0; i < _subset.size(); ++i) {
        buffer[_subset[i]] = v[i] * weights[i];
    }
    return _mat->cmul_safe(j, _mask, buffer);
}

ADELIE_CORE_MATRIX_NAIVE_RSUBSET_TP
void
ADELIE_CORE_MATRIX_NAIVE_RSUBSET::ctmul(
    int j, 
    value_t v, 
    Eigen::Ref<vec_value_t> out
)
{
    base_t::check_ctmul(j, out.size(), rows(), cols());
    _buffer.setZero();
    _mat->ctmul(j, v, _buffer);
    for (int i = 0; i < _subset.size(); ++i) {
        out[i] += _buffer[_subset[i]];
    }
}

ADELIE_CORE_MATRIX_NAIVE_RSUBSET_TP
void
ADELIE_CORE_MATRIX_NAIVE_RSUBSET::bmul(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
)
{
    base_t::check_bmul(j, q, v.size(), weights.size(), out.size(), rows(), cols());
    _buffer.setZero();
    for (int i = 0; i < _subset.size(); ++i) {
        _buffer[_subset[i]] = v[i] * weights[i];
    }
    _mat->bmul(j, q, _mask, _buffer, out);
}

ADELIE_CORE_MATRIX_NAIVE_RSUBSET_TP
void
ADELIE_CORE_MATRIX_NAIVE_RSUBSET::bmul_safe(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
{
    base_t::check_bmul(j, q, v.size(), weights.size(), out.size(), rows(), cols());
    vec_value_t buffer(_mat->rows());
    buffer.setZero();
    for (int i = 0; i < _subset.size(); ++i) {
        buffer[_subset[i]] = v[i] * weights[i];
    }
    _mat->bmul_safe(j, q, _mask, buffer, out);
}

ADELIE_CORE_MATRIX_NAIVE_RSUBSET_TP
void
ADELIE_CORE_MATRIX_NAIVE_RSUBSET::btmul(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    Eigen::Ref<vec_value_t> out
)
{
    base_t::check_btmul(j, q, v.size(), out.size(), rows(), cols());
    _buffer.setZero();
    _mat->btmul(j, q, v, _buffer);
    for (int i = 0; i < _subset.size(); ++i) {
        out[i] += _buffer[_subset[i]];
    }
}

ADELIE_CORE_MATRIX_NAIVE_RSUBSET_TP
void
ADELIE_CORE_MATRIX_NAIVE_RSUBSET::mul(
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
{
    vec_value_t buffer(_mat->rows());
    buffer.setZero();
    for (int i = 0; i < _subset.size(); ++i) {
        buffer[_subset[i]] = v[i] * weights[i];
    }
    _mat->mul(_mask, buffer, out);
}

ADELIE_CORE_MATRIX_NAIVE_RSUBSET_TP
int
ADELIE_CORE_MATRIX_NAIVE_RSUBSET::rows() const
{
    return _subset.size();
}

ADELIE_CORE_MATRIX_NAIVE_RSUBSET_TP
int
ADELIE_CORE_MATRIX_NAIVE_RSUBSET::cols() const
{
    return _mat->cols();
}

ADELIE_CORE_MATRIX_NAIVE_RSUBSET_TP
void
ADELIE_CORE_MATRIX_NAIVE_RSUBSET::cov(
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
    vec_value_t buffer(_mat->rows());
    buffer.setZero();
    for (int i = 0; i < _subset.size(); ++i) {
        buffer[_subset[i]] = sqrt_weights[i];
    }
    _mat->cov(j, q, buffer, out);
}

ADELIE_CORE_MATRIX_NAIVE_RSUBSET_TP
void
ADELIE_CORE_MATRIX_NAIVE_RSUBSET::sq_mul(
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
{
    vec_value_t buffer(_mat->rows());
    buffer.setZero();
    for (int i = 0; i < _subset.size(); ++i) {
        buffer[_subset[i]] = weights[i];
    }
    _mat->sq_mul(buffer, out);
}

ADELIE_CORE_MATRIX_NAIVE_RSUBSET_TP
void
ADELIE_CORE_MATRIX_NAIVE_RSUBSET::sp_tmul(
    const sp_mat_value_t& v, 
    Eigen::Ref<rowmat_value_t> out
) const
{
    base_t::check_sp_tmul(
        v.rows(), v.cols(), out.rows(), out.cols(), rows(), cols()
    );
    rowmat_value_t _out(out.rows(), _mat->rows());
    _mat->sp_tmul(v, _out);
    for (int i = 0; i < _subset.size(); ++i) {
        out.col(i) = _out.col(_subset[i]);
    }
}

ADELIE_CORE_MATRIX_NAIVE_RSUBSET_TP
void
ADELIE_CORE_MATRIX_NAIVE_RSUBSET::mean(
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
{
    vec_value_t new_w(_mat->rows());
    new_w.setZero();
    for (int i = 0; i < _subset.size(); ++i) {
        new_w[_subset[i]] = weights[i];
    }
    _mat->mean(new_w, out);
}

ADELIE_CORE_MATRIX_NAIVE_RSUBSET_TP
void
ADELIE_CORE_MATRIX_NAIVE_RSUBSET::var(
    const Eigen::Ref<const vec_value_t>& centers,
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
{
    vec_value_t new_w(_mat->rows());
    new_w.setZero();
    for (int i = 0; i < _subset.size(); ++i) {
        new_w[_subset[i]] = weights[i];
    }
    _mat->var(centers, new_w, out);
}

} // namespace matrix
} // namespace adelie_core