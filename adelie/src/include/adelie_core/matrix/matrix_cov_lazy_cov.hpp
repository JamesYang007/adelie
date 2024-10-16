#pragma once
#include <adelie_core/matrix/matrix_cov_base.hpp>

namespace adelie_core {
namespace matrix {

template <class DenseType, class IndexType=Eigen::Index>
class MatrixCovLazyCov: public MatrixCovBase<typename std::decay_t<DenseType>::Scalar, IndexType>
{
public: 
    using base_t = MatrixCovBase<typename std::decay_t<DenseType>::Scalar, IndexType>;
    using typename base_t::value_t;
    using typename base_t::index_t;
    using typename base_t::vec_index_t;
    using typename base_t::vec_value_t;
    using typename base_t::colmat_value_t;
    using dense_t = DenseType;
    
private:
    const Eigen::Map<const dense_t> _X;     // ref of data matrix
    const size_t _n_threads;                // number of threads
    std::vector<util::rowmat_type<value_t>> _cache; // cache of covariance slices
    std::vector<index_t> _index_map; // map feature i to index of _cache
    std::vector<index_t> _slice_map; // map feature i to slice of _cache[_index_map[i]]

    inline void cache(int i, int p);

public: 
    explicit MatrixCovLazyCov(
        const Eigen::Ref<const dense_t>& X,
        size_t n_threads
    ); 

    using base_t::rows;

    void bmul(
        const Eigen::Ref<const vec_index_t>& subset,
        const Eigen::Ref<const vec_index_t>& indices,
        const Eigen::Ref<const vec_value_t>& values,
        Eigen::Ref<vec_value_t> out
    ) override;

    void mul(
        const Eigen::Ref<const vec_index_t>& indices,
        const Eigen::Ref<const vec_value_t>& values,
        Eigen::Ref<vec_value_t> out
    ) override;

    void to_dense(
        int i, int p,
        Eigen::Ref<colmat_value_t> out
    ) override;

    int cols() const override; 
};

} // namespace matrix
} // namespace adelie_core