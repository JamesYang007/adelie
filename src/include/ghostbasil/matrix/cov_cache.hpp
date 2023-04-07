#pragma once
#include <vector>
#include <Eigen/Core>
#include <ghostbasil/util/types.hpp>
#include <ghostbasil/matrix/forward_decl.hpp>

namespace ghostbasil {

template <class CovCacheType>
class CovCacheRow
{
public:
    using cov_cache_t = CovCacheType;
    using index_t = typename cov_cache_t::index_t;
    
private:
    const cov_cache_t& cov_cache_;
    index_t i_;

public:
    CovCacheRow(
        const cov_cache_t& cov_cache,
        index_t i
    )
        : cov_cache_(cov_cache),
          i_(i)
    {}
    
    auto segment(index_t j, index_t q) const
    {
        const auto& mat = cov_cache_.get_cache(j);
        return mat.row(i_).head(q);
    }
};

template <class XType, class ValueType>
class CovCache
{
public: 
    using x_t = XType;
    using value_t = ValueType;
    using index_t = Eigen::Index;
    
private:
    const x_t& X_;  // ref of data matrix
    std::vector<util::mat_type<value_t>> cache_; // cache of covariance slices
    std::vector<index_t> index_map_; // map group i to index into cache_

public: 
    CovCache(const x_t& X) 
        : X_(X),
          cache_(),
          index_map_(X.cols())
    {
        cache_.reserve(X.cols());
    }
    
    auto cols() const { return X_.cols(); }
    auto rows() const { return cols(); }
    
    void cache(index_t j, index_t q) 
    {
        const auto next_idx = cache_.size();
        index_map_[j] = next_idx;
        cache_.emplace_back(X_.transpose() * X_.block(0, j, X_.rows(), q));
    }
    
    const auto& get_cache(index_t j) const
    {
        return cache_[index_map_[j]];
    }

    auto block(index_t i, index_t j, index_t p, index_t q) const
    {
        const auto& mat = get_cache(j);
        return mat.block(i, 0, p, q);
    }
    
    template <index_t p, index_t q>
    auto block(index_t i, index_t j) const
    {
        const auto& mat = get_cache(j);
        return mat.template block<p, q>(i, 0);
    }
    
    auto col(index_t j) const
    {
        const auto& mat = get_cache(j);
        assert(mat.cols() == 1);
        return mat.col(0);
    }
    
    auto row(index_t i) const
    {
        return CovCacheRow<CovCache>(*this, i);
    }
};

} // namespace ghostbasil