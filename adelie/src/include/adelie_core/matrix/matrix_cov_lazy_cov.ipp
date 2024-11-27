#pragma once
#include <vector>
#include <adelie_core/matrix/matrix_cov_lazy_cov.hpp>
#include <adelie_core/matrix/utils.hpp>

namespace adelie_core {
namespace matrix {

ADELIE_CORE_MATRIX_COV_LAZY_COV_TP
void
ADELIE_CORE_MATRIX_COV_LAZY_COV::cache(
    int i, 
    int p
) 
{
    const auto next_idx = _cache.size();

    // populate maps
    for (int k = 0; k < p; ++k) {
        _index_map[i + k] = next_idx;
        _slice_map[i + k] = k;
    }

    const auto block = _X.middleCols(i, p);
    util::rowmat_type<value_t> cov(p, _X.cols());
    if (_n_threads <= 1 || util::omp_in_parallel()) { 
        Eigen::setNbThreads(1);
        cov.noalias() = block.transpose() * _X; 
        _cache.emplace_back(std::move(cov));
        return;
    }
    const int n_blocks = std::min<size_t>(_n_threads, p);
    const int block_size = p / n_blocks;
    const int remainder = p % n_blocks;
    #pragma omp parallel for schedule(static) num_threads(_n_threads)
    for (int t = 0; t < n_blocks; ++t) {
        const auto begin = (
            std::min<int>(t, remainder) * (block_size + 1) 
            + std::max<int>(t-remainder, 0) * block_size
        );
        const auto size = block_size + (t < remainder);
        Eigen::setNbThreads(1);
        cov.middleRows(begin, size).noalias() = (
            block.transpose().middleRows(begin, size) * _X
        );
    }
    _cache.emplace_back(std::move(cov));
}

ADELIE_CORE_MATRIX_COV_LAZY_COV_TP
ADELIE_CORE_MATRIX_COV_LAZY_COV::MatrixCovLazyCov(
    const Eigen::Ref<const dense_t>& X,
    size_t n_threads
): 
    _X(X.data(), X.rows(), X.cols()),
    _n_threads(n_threads),
    _index_map(X.cols(), -1),
    _slice_map(X.cols(), -1)
{
    if (n_threads < 1) {
        throw util::adelie_core_error("n_threads must be >= 1.");
    }
    _cache.reserve(X.cols());
}

ADELIE_CORE_MATRIX_COV_LAZY_COV_TP
void
ADELIE_CORE_MATRIX_COV_LAZY_COV::bmul(
    const Eigen::Ref<const vec_index_t>& subset,
    const Eigen::Ref<const vec_index_t>& indices,
    const Eigen::Ref<const vec_value_t>& values,
    Eigen::Ref<vec_value_t> out
) 
{
    base_t::check_bmul(subset.size(), indices.size(), values.size(), out.size(), rows(), cols());
    // cache first
    for (int i_idx = 0; i_idx < indices.size(); ++i_idx) {
        const auto i = indices[i_idx];
        if (_index_map[i] < 0) {
            int cache_size = 0;
            for(; i+cache_size < cols() && _index_map[i+cache_size] < 0 && 
                    indices[i_idx+cache_size] == i+cache_size; ++cache_size);
            cache(i, cache_size);
        }
    }

    // update output
    out.setZero();
    for (int j_idx = 0; j_idx < subset.size(); ++j_idx) {
        const auto j = subset[j_idx];
        for (int i_idx = 0; i_idx < indices.size(); ++i_idx) {
            const auto i = indices[i_idx];
            const auto& mat = _cache[_index_map[i]];
            const auto i_rel = _slice_map[i];
            const auto v = values[i_idx];
            out[j_idx] += v * mat(i_rel, j);
        }
    }
}

ADELIE_CORE_MATRIX_COV_LAZY_COV_TP
void
ADELIE_CORE_MATRIX_COV_LAZY_COV::mul(
    const Eigen::Ref<const vec_index_t>& indices,
    const Eigen::Ref<const vec_value_t>& values,
    Eigen::Ref<vec_value_t> out
) const
{
    base_t::check_mul(indices.size(), values.size(), out.size(), rows(), cols());
    const auto n = _X.rows();
    const auto p = _X.cols();
    const auto max_np = std::max(n, p);
    util::rowmat_type<value_t> buff;
    vec_value_t vbuff;
    out.setZero();
    for (int i_idx = 0; i_idx < indices.size();) {
        const auto i = indices[i_idx];
        if (_index_map[i] < 0) {
            int block_size = 0;
            for(; i+block_size < p && _index_map[i+block_size] < 0 && 
                    indices[i_idx+block_size] == i+block_size; ++block_size);
            if (static_cast<size_t>(buff.size()) < _n_threads * max_np) {
                buff.resize(_n_threads * (_n_threads > 1) * !util::omp_in_parallel(), max_np);
            }
            if (vbuff.size() < n+p) vbuff.resize(n+p);
            auto Xv_m = vbuff.head(n).matrix();
            dgemv(
                _X.middleCols(i, block_size).transpose(),
                values.segment(i_idx, block_size).matrix(),
                _n_threads,
                buff,
                Xv_m
            );
            auto XTXv_m = vbuff.tail(p).matrix();
            dgemv(
                _X,
                Xv_m,
                _n_threads,
                buff,
                XTXv_m
            );
            dvaddi(out, XTXv_m.array(), _n_threads);
            i_idx += block_size;
            continue;
        }
        const auto& mat = _cache[_index_map[i]];
        const auto i_rel = _slice_map[i];
        const auto v = values[i_idx];
        dvaddi(out, v * mat.row(i_rel).array(), _n_threads);
        ++i_idx;
    }
}

ADELIE_CORE_MATRIX_COV_LAZY_COV_TP
void
ADELIE_CORE_MATRIX_COV_LAZY_COV::to_dense(
    int i, int p,
    Eigen::Ref<colmat_value_t> out
) const
{
    base_t::check_to_dense(i, p, out.rows(), out.cols(), rows(), cols());
    const auto X_block = _X.middleCols(i, p);
    int n_processed = 0;
    while (n_processed < p) {
        const auto k = i + n_processed;
        if (_index_map[k] < 0) {
            int block_size = 0;
            for(; k+block_size < i+p && _index_map[k+block_size] < 0; ++block_size);
            const auto Xk = _X.middleCols(k, block_size);
            auto out_m = out.middleCols(n_processed, block_size);
            out_m.noalias() = X_block.transpose() * Xk;
            n_processed += block_size;
            continue;
        }
        const auto& mat = _cache[_index_map[k]];
        const auto k_rel = _slice_map[k];
        const auto size = std::min<size_t>(mat.rows()-k_rel, p-n_processed);
        out.middleCols(n_processed, size) = mat.block(k_rel, i, size, p).transpose();
        n_processed += size;
    }
}

ADELIE_CORE_MATRIX_COV_LAZY_COV_TP
int
ADELIE_CORE_MATRIX_COV_LAZY_COV::cols() const
{ 
    return _X.cols(); 
}

} // namespace matrix
} // namespace adelie_core