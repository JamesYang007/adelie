#pragma once
#include <vector>
#include <adelie_core/util/types.hpp>
#include <adelie_core/matrix/matrix_cov_base.hpp>
#include <adelie_core/matrix/utils.hpp>

namespace adelie_core {
namespace matrix {

template <class DenseType>
class MatrixCovLazy: 
    public MatrixCovBase<typename std::decay_t<DenseType>::Scalar>
{
public: 
    using base_t = MatrixCovBase<typename std::decay_t<DenseType>::Scalar>;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using typename base_t::colmat_value_t;
    using dense_t = DenseType;
    using index_t = Eigen::Index;
    
private:
    const Eigen::Map<const dense_t> _X;     // ref of data matrix
    const size_t _n_threads;                // number of threads
    std::vector<util::rowmat_type<value_t>> _cache; // cache of covariance slices
    std::vector<index_t> _index_map; // map feature i to index of _cache
    std::vector<index_t> _slice_map; // map feature i to slice of _cache[_index_map[i]]

    void cache(int i, int p) 
    {
        const auto next_idx = _cache.size();

        // populate maps
        for (int k = 0; k < p; ++k) {
            _index_map[i + k] = next_idx;
            _slice_map[i + k] = k;
        }

        const auto block = _X.middleCols(i, p);

        const int n_blocks = std::min<size_t>(_n_threads, p);
        const int block_size = p / n_blocks;
        const int remainder = p % n_blocks;
        util::rowmat_type<value_t> cov(p, _X.cols());
        #pragma omp parallel for schedule(static) num_threads(n_blocks)
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

public: 
    MatrixCovLazy(
        const Eigen::Ref<const dense_t>& X,
        size_t n_threads
    ): 
        _X(X.data(), X.rows(), X.cols()),
        _n_threads(n_threads),
        _cache(),
        _index_map(X.cols(), -1),
        _slice_map(X.cols(), -1)
    {
        _cache.reserve(X.cols());
    }

    void bmul(
        int i, int j, int p, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {
        if (i < 0 || j < 0 || p <= 0 || q <= 0) {
            throw std::runtime_error(
                "Indices must be all non-negative and sizes must be all positive."
            );
        }
        const index_t ci = _index_map[i];
        if (ci < 0) {
            cache(i, p);
        }
        if (_index_map[i] != _index_map[i + p - 1]) {
            throw std::runtime_error(
                "Rows i,..., i+p-1 must be in the same cached block."
            );
        }
        const auto& mat = _cache[_index_map[i]];
        out.matrix().noalias() = v.matrix() * mat.block(_slice_map[i], j, p, q);
    }

    void to_dense(
        int i, int j, int p, int q,
        Eigen::Ref<colmat_value_t> out
    ) const override
    {
        if (i < 0 || i > static_cast<int>(_index_map.size())) {
            throw std::runtime_error(
                "Index is out of range."
            );
        }
        const index_t ci = _index_map[i];
        if (ci < 0) {
            const auto Xi = _X.middleCols(i, p);
            const auto Xj = _X.middleCols(j, q);
            out.noalias() = Xi.transpose() * Xj;
            return;
        }
        const auto& mat = _cache[ci];
        out.noalias() = mat.block(_slice_map[i], j, p, q);
    }

    int cols() const override { return _X.cols(); }
};

} // namespace matrix
} // namespace adelie_core