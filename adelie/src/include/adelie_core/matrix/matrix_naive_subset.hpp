#pragma once
#include <adelie_core/matrix/matrix_naive_base.hpp>

namespace adelie_core {
namespace matrix {

template <class ValueType, class IndexType=Eigen::Index>
class MatrixNaiveCSubset: public MatrixNaiveBase<ValueType, IndexType>
{
public:
    using base_t = MatrixNaiveBase<ValueType, IndexType>;
    using typename base_t::value_t;
    using typename base_t::index_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::colmat_value_t;
    using typename base_t::rowmat_value_t;
    using typename base_t::sp_mat_value_t;
    using dyn_vec_index_t = std::vector<index_t>;
    using map_cvec_index_t = Eigen::Map<const vec_index_t>;
    
private:
    base_t* _mat;               // underlying matrix
    const map_cvec_index_t _subset;  // column subset
    const std::tuple<
        vec_index_t,
        dyn_vec_index_t
    > _subset_cinfo;            // 1) number of elements left
                                // in the contiguous sub-chunk
                                // starting at _subset[i]
                                // 2) beginning index to each
                                // contiguous sub-chunk
    const size_t _n_threads;

    static inline auto init_subset_cinfo(
        const Eigen::Ref<const vec_index_t>& subset
    );

public:
    explicit MatrixNaiveCSubset(
        base_t& mat,
        const Eigen::Ref<const vec_index_t>& subset,
        size_t n_threads
    );

    value_t cmul(
        int j, 
        const Eigen::Ref<const vec_value_t>& v,
        const Eigen::Ref<const vec_value_t>& weights
    ) override;

    void ctmul(
        int j, 
        value_t v, 
        Eigen::Ref<vec_value_t> out
    ) override;

    void bmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override;

    void btmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
    ) override;

    void mul(
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override;

    int rows() const override;
    
    int cols() const override;

    void cov(
        int j, int q,
        const Eigen::Ref<const vec_value_t>& sqrt_weights,
        Eigen::Ref<colmat_value_t> out,
        Eigen::Ref<colmat_value_t> buffer
    ) override;

    void sp_tmul(
        const sp_mat_value_t& v, 
        Eigen::Ref<rowmat_value_t> out
    ) override;
};

template <class ValueType, class IndexType=Eigen::Index>
class MatrixNaiveRSubset: public MatrixNaiveBase<ValueType, IndexType>
{
public:
    using base_t = MatrixNaiveBase<ValueType, IndexType>;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::colmat_value_t;
    using typename base_t::rowmat_value_t;
    using typename base_t::sp_mat_value_t;
    using vec_bool_t = util::rowvec_type<bool>;
    using map_cvec_index_t = Eigen::Map<const vec_index_t>;
    
private:
    base_t* _mat;               // underlying matrix
    const map_cvec_index_t _subset;
    const vec_value_t _mask;  
    const size_t _n_threads;
    vec_value_t _buffer;
    vec_value_t _cov_buffer;

    static inline auto init_mask(
        size_t n,
        const Eigen::Ref<const vec_index_t>& subset
    );
    
public:
    explicit MatrixNaiveRSubset(
        base_t& mat,
        const Eigen::Ref<const vec_index_t>& subset,
        size_t n_threads
    );

    value_t cmul(
        int j, 
        const Eigen::Ref<const vec_value_t>& v,
        const Eigen::Ref<const vec_value_t>& weights
    ) override;

    void ctmul(
        int j, 
        value_t v, 
        Eigen::Ref<vec_value_t> out
    ) override;

    void bmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override;

    void btmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
    ) override;

    void mul(
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override;

    int rows() const override;
    
    int cols() const override;

    void cov(
        int j, int q,
        const Eigen::Ref<const vec_value_t>& sqrt_weights,
        Eigen::Ref<colmat_value_t> out,
        Eigen::Ref<colmat_value_t> buffer
    ) override;

    void sp_tmul(
        const sp_mat_value_t& v, 
        Eigen::Ref<rowmat_value_t> out
    ) override;
};

} // namespace matrix
} // namespace adelie_core