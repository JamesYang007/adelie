#pragma once
#include <vector>
#include <adelie_core/matrix/matrix_cov_lazy_cov.hpp>
#include <adelie_core/matrix/utils.hpp>

namespace adelie_core {
namespace matrix {

template <class DenseType, class IndexType>
void
MatrixCovLazyCov<DenseType, IndexType>::cache(
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
    if (_n_threads <= 1) { 
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
        cov.middleRows(begin, size).noalias() = (
            block.transpose().middleRows(begin, size) * _X
        );
    }
    _cache.emplace_back(std::move(cov));
}

template <class DenseType, class IndexType>
MatrixCovLazyCov<DenseType, IndexType>::MatrixCovLazyCov(
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

template <class DenseType, class IndexType>
void
MatrixCovLazyCov<DenseType, IndexType>::bmul(
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

template <class DenseType, class IndexType>
void
MatrixCovLazyCov<DenseType, IndexType>::mul(
    const Eigen::Ref<const vec_index_t>& indices,
    const Eigen::Ref<const vec_value_t>& values,
    Eigen::Ref<vec_value_t> out
) 
{
    base_t::check_mul(indices.size(), values.size(), out.size(), rows(), cols());
    out.setZero();
    for (int i_idx = 0; i_idx < indices.size(); ++i_idx) {
        const auto i = indices[i_idx];
        if (_index_map[i] < 0) {
            int cache_size = 0;
            for(; i+cache_size < cols() && _index_map[i+cache_size] < 0 && 
                    indices[i_idx+cache_size] == i+cache_size; ++cache_size);
            cache(i, cache_size);
        }
        const auto& mat = _cache[_index_map[i]];
        const auto i_rel = _slice_map[i];
        const auto v = values[i_idx];
        dvaddi(out, v * mat.row(i_rel).array(), _n_threads);
    }
}

template <class DenseType, class IndexType>
void
MatrixCovLazyCov<DenseType, IndexType>::to_dense(
    int i, int p,
    Eigen::Ref<colmat_value_t> out
) 
{
    base_t::check_to_dense(i, p, out.rows(), out.cols(), rows(), cols());
    int n_processed = 0;
    while (n_processed < p) {
        const auto k = i + n_processed;
        if (_index_map[k] < 0) {
            int cache_size = 0;
            for(; k+cache_size < i+p && _index_map[k+cache_size] < 0; ++cache_size);
            cache(k, cache_size);
        }
        const auto& mat = _cache[_index_map[k]];
        const auto k_rel = _slice_map[k];
        const auto size = std::min<size_t>(mat.rows()-k_rel, p-n_processed);
        out.middleCols(n_processed, size) = mat.block(k_rel, i, size, p).transpose();
        n_processed += size;
    }
}

template <class DenseType, class IndexType>
int
MatrixCovLazyCov<DenseType, IndexType>::cols() const
{ 
    return _X.cols(); 
}

} // namespace matrix
} // namespace adelie_core