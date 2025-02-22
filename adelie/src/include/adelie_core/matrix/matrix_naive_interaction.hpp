#pragma once
#include <adelie_core/matrix/matrix_naive_base.hpp>

#ifndef ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE_TP
#define ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE_TP \
    template <class DenseType, class IndexType>
#endif
#ifndef ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE
#define ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE \
    MatrixNaiveInteractionDense<DenseType, IndexType>
#endif

namespace adelie_core {
namespace matrix {

template <class DenseType, class IndexType=Eigen::Index>
class MatrixNaiveInteractionDense: public MatrixNaiveBase<typename DenseType::Scalar, IndexType>
{
public:
    using base_t = MatrixNaiveBase<typename DenseType::Scalar, IndexType>;
    using typename base_t::value_t;
    using typename base_t::index_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::colmat_value_t;
    using typename base_t::rowmat_value_t;
    using typename base_t::sp_mat_value_t;
    using dense_t = DenseType;
    using rowarr_index_t = util::rowarr_type<index_t>;

private:
    static constexpr size_t _n_levels_cont = 2;

    const Eigen::Map<const dense_t> _mat;   // (n, d) underlying matrix
    const Eigen::Map<const rowarr_index_t> _pairs;  // (G, 2) pair matrix
    const Eigen::Map<const vec_index_t> _levels;  // (d,) number of levels
    const vec_index_t _outer;               // (G+1,) outer vector
    const size_t _cols;                     // number of columns (p)
    const vec_index_t _slice_map;           // (p,) array mapping to matrix slice
    const vec_index_t _index_map;           // (p,) array mapping to (relative) index of the slice
    const size_t _n_threads;                // number of threads
    vec_value_t _buff;

    static inline auto init_outer(
        const Eigen::Ref<const rowarr_index_t>& pairs,
        const Eigen::Ref<const vec_index_t>& levels
    );

    static inline auto init_slice_map(
        const Eigen::Ref<const rowarr_index_t>& pairs,
        const Eigen::Ref<const vec_index_t>& levels,
        size_t cols
    );

    static inline auto init_index_map(
        const Eigen::Ref<const rowarr_index_t>& pairs,
        const Eigen::Ref<const vec_index_t>& levels,
        size_t cols
    );

    inline value_t _cmul(
        int j, 
        const Eigen::Ref<const vec_value_t>& v,
        const Eigen::Ref<const vec_value_t>& weights,
        size_t n_threads,
        Eigen::Ref<vec_value_t> buff
    ) const;

    inline void _ctmul(
        int j, 
        value_t v, 
        Eigen::Ref<vec_value_t> out,
        size_t n_threads
    ) const;

    inline void _bmul(
        int begin,
        int i0, int i1,
        int l0, int l1,
        int index,
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out,
        Eigen::Ref<vec_value_t> buff,
        size_t n_threads
    ) const;

    inline void _sq_bmul(
        int i0, int i1,
        int l0, int l1,
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out,
        Eigen::Ref<vec_value_t> buff
    ) const;

    inline void _btmul(
        int begin,
        int i0, int i1,
        int l0, int l1,
        int index,
        size_t size,
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out,
        size_t n_threads
    ) const;

public:
    explicit MatrixNaiveInteractionDense(
        const Eigen::Ref<const dense_t>& mat,
        const Eigen::Ref<const rowarr_index_t>& pairs,
        const Eigen::Ref<const vec_index_t>& levels,
        size_t n_threads
    );

    vec_index_t groups() const
    {
        const size_t G = _outer.size() - 1;
        return _outer.head(G);
    }

    vec_index_t group_sizes() const
    {
        const size_t G = _outer.size() - 1;
        return _outer.tail(G) - _outer.head(G);
    }

    ADELIE_CORE_MATRIX_NAIVE_PURE_OVERRIDE_DECL
    ADELIE_CORE_MATRIX_NAIVE_OVERRIDE_DECL
};

} // namespace matrix 
} // namespace adelie_core