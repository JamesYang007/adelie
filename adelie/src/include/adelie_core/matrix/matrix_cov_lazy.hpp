#pragma once
#include <vector>
#include <adelie_core/util/types.hpp>
#include <adelie_core/matrix/matrix_cov_base.hpp>
#include <adelie_core/matrix/utils.hpp>

namespace adelie_core {
namespace matrix {

template <class DenseType>
class MatrixCovLazy: public MatrixCovBase<typename std::decay_t<DenseType>::Scalar>
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
    util::rowmat_type<value_t> _buff;
    util::rowvec_type<value_t> _vbuff;

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

public: 
    explicit MatrixCovLazy(
        const Eigen::Ref<const dense_t>& X,
        size_t n_threads
    ): 
        _X(X.data(), X.rows(), X.cols()),
        _n_threads(n_threads),
        _cache(),
        _index_map(X.cols(), -1),
        _slice_map(X.cols(), -1),
        _buff(_n_threads, X.cols()),
        _vbuff(X.cols())
    {
        _cache.reserve(X.cols());
    }

    using base_t::rows;

    void bmul(
        int i, int j, int p, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_bmul(i, j, p, q, v.size(), out.size(), rows(), cols());
        Eigen::Map<vec_value_t> vbuff(_vbuff.data(), q);
        int n_processed = 0;
        out.setZero();
        while (n_processed < p) {
            const auto k = i + n_processed;
            const auto ck = _index_map[k];
            if (ck < 0) {
                int cache_size = 0;
                for(; k+cache_size < cols() && _index_map[k+cache_size] < 0; ++cache_size);
                cache(k, cache_size);
            }
            const auto& mat = _cache[_index_map[k]];
            const auto size = std::min<size_t>(mat.rows(), p-n_processed);
            vbuff = v.matrix().segment(n_processed, size) * mat.block(_slice_map[k], j, size, q);
            out += vbuff;
            n_processed += size;
        }
    }

    void mul(
        int i, int p,
        const Eigen::Ref<const vec_value_t>& v,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_mul(i, p, v.size(), out.size(), rows(), cols());
        Eigen::Map<vec_value_t> vbuff(_vbuff.data(), cols());
        int n_processed = 0;
        out.setZero();
        while (n_processed < p) {
            const auto k = i + n_processed;
            const auto ck = _index_map[k];
            if (ck < 0) {
                int cache_size = 0;
                for(; k+cache_size < cols() && _index_map[k+cache_size] < 0; ++cache_size);
                cache(k, cache_size);
            }
            const auto& mat = _cache[_index_map[k]];
            const auto size = std::min<size_t>(mat.rows(), p-n_processed);
            auto vbuffm = vbuff.matrix();
            dgemv(
                mat.middleRows(_slice_map[k], size),
                v.matrix().segment(n_processed, size),
                _n_threads,
                _buff,
                vbuffm
            );
            dvaddi(out, vbuff, _n_threads);
            n_processed += size;
        }
    }

    void to_dense(
        int i, int p,
        Eigen::Ref<colmat_value_t> out
    ) override
    {
        base_t::check_to_dense(i, p, out.rows(), out.cols(), rows(), cols());
        int n_processed = 0;
        while (n_processed < p) {
            const auto k = i + n_processed;
            const auto ck = _index_map[k];
            if (ck < 0) {
                int cache_size = 0;
                for(; k+cache_size < cols() && _index_map[k+cache_size] < 0; ++cache_size);
                cache(k, cache_size);
            }
            const auto& mat = _cache[_index_map[k]];
            const auto size = std::min<size_t>(mat.rows(), p-n_processed);
            out.middleRows(n_processed, size) = mat.block(_slice_map[k], i, size, p);
            n_processed += size;
        }
    }

    int cols() const override { return _X.cols(); }
};

} // namespace matrix
} // namespace adelie_core