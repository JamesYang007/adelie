#pragma once
#include <adelie_core/matrix/matrix_naive_base.hpp>

#ifndef ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_DENSE_TP
#define ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_DENSE_TP \
    template <class DenseType, class MaskType, class IndexType>
#endif
#ifndef ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_DENSE
#define ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_DENSE \
    MatrixNaiveConvexGatedReluDense<DenseType, MaskType, IndexType>
#endif

#ifndef ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_SPARSE_TP
#define ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_SPARSE_TP \
    template <class SparseType, class MaskType, class IndexType>
#endif
#ifndef ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_SPARSE
#define ADELIE_CORE_MATRIX_NAIVE_CONVEX_GATED_RELU_SPARSE \
    MatrixNaiveConvexGatedReluSparse<SparseType, MaskType, IndexType>
#endif

namespace adelie_core {
namespace matrix {

template <
    class DenseType, 
    class MaskType,
    class IndexType=Eigen::Index
>
class MatrixNaiveConvexGatedReluDense: public MatrixNaiveBase<typename DenseType::Scalar, IndexType>
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
    using mask_t = MaskType;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;

private:
    const Eigen::Map<const dense_t> _mat;      // (n, d) underlying matrix
    const Eigen::Map<const mask_t> _mask;      // (n, m) mask matrix
    const size_t _n_threads;
    vec_value_t _buff;

    inline auto _cmul(
        int j, 
        const Eigen::Ref<const vec_value_t>& v,
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> buff
    ) const;

    inline void _ctmul(
        int j, 
        value_t v, 
        Eigen::Ref<vec_value_t> out,
        size_t n_threads
    ) const;

    inline void _bmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out,
        Eigen::Ref<vec_value_t> buffer
    ) const;

public:
    explicit MatrixNaiveConvexGatedReluDense(
        const Eigen::Ref<const dense_t>& mat,
        const Eigen::Ref<const mask_t>& mask,
        size_t n_threads
    );

    ADELIE_CORE_MATRIX_NAIVE_PURE_OVERRIDE_DECL
};

template <
    class SparseType, 
    class MaskType,
    class IndexType=Eigen::Index
>
class MatrixNaiveConvexGatedReluSparse: public MatrixNaiveBase<typename SparseType::Scalar, IndexType>
{
public:
    using base_t = MatrixNaiveBase<typename SparseType::Scalar, IndexType>;
    using typename base_t::value_t;
    using typename base_t::index_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::colmat_value_t;
    using typename base_t::rowmat_value_t;
    using typename base_t::sp_mat_value_t;
    using sparse_t = SparseType;
    using mask_t = MaskType;
    using vec_sp_value_t = vec_value_t;
    using vec_sp_index_t = util::rowvec_type<typename sparse_t::StorageIndex>;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;

    static_assert(!sparse_t::IsRowMajor, "MatrixNaiveConvexGatedReluSparse: only column-major allowed!");

private:
    const Eigen::Map<const sparse_t> _mat;      // (n, d) underlying matrix
    const Eigen::Map<const mask_t> _mask;      // (n, m) mask matrix
    const size_t _n_threads;
    vec_value_t _buff;

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
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out,
        Eigen::Ref<vec_value_t> buff
    ) const;

public:
    explicit MatrixNaiveConvexGatedReluSparse(
        size_t rows,
        size_t cols,
        size_t nnz,
        const Eigen::Ref<const vec_sp_index_t>& outer,
        const Eigen::Ref<const vec_sp_index_t>& inner,
        const Eigen::Ref<const vec_sp_value_t>& value,
        const Eigen::Ref<const mask_t>& mask,
        size_t n_threads
    );

    ADELIE_CORE_MATRIX_NAIVE_PURE_OVERRIDE_DECL
};

} // namespace matrix
} // namespace adelie_core 