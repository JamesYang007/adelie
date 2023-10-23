#pragma once
#include <adelie_core/util/types.hpp>

namespace adelie_core {
namespace matrix {

template <class ValueType, class IndexType=int> 
class MatrixNaiveBase
{
public:
    using value_t = ValueType;
    using index_t = IndexType;
    using vec_value_t = util::rowvec_type<value_t>;
    using vec_index_t = util::rowvec_type<index_t>;
    using colmat_value_t = util::colmat_type<value_t>;
    using rowmat_value_t = util::rowmat_type<value_t>;
    using sp_mat_value_t = Eigen::SparseMatrix<value_t, Eigen::RowMajor>;
    
    virtual ~MatrixNaiveBase() {}
    
    /**
     * @brief Computes v^T X[:, j] where X is the current matrix.
     * 
     * @param   j       column index.
     * @param   v       vector to multiply with.  
     * @return  resulting value.
     */
    virtual value_t cmul(
        int j, 
        const Eigen::Ref<const vec_value_t>& v
    ) const =0;

    /**
     * @brief Computes v X[:, j]^T where X is the current matrix.
     * 
     * @param j         column index. 
     * @param v         scalar to multiply with. 
     * @param out       resulting row vector.
     */
    virtual void ctmul(
        int j, 
        value_t v, 
        Eigen::Ref<vec_value_t> out
    ) const =0;

    /**
     * @brief Computes v^T X[:, j:j+q] where X is the current matrix.
     * 
     * @param j     begin column index. 
     * @param q     number of columns.
     * @param v     vector to multiply with.
     * @param out   resulting row vector.
     */
    virtual void bmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
    ) =0;

    /**
     * @brief Computes v^T X[:, j:j+q]^T where X is the current matrix.
     * 
     * @param j     begin column index. 
     * @param q     number of columns.
     * @param v     vector to multiply with.
     * @param out   resulting row vector.
     */
    virtual void btmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
    ) =0;

    /**
     * @brief Computes v X[:, j:j+q]^T where X is the current matrix.
     * 
     * @param j     begin column index. 
     * @param q     number of columns.
     * @param v     (l, p) sparse matrix to multiply with.
     * @param out   (l, n) resulting row vector.
     */
    virtual void sp_btmul(
        int j, int q,
        const sp_mat_value_t& v,
        Eigen::Ref<rowmat_value_t> out
    ) const =0;

    /**
     * @brief Computes the squared norm of a column of the matrix.
     * 
     * @param j     column index.
     * @param q     number of columns.
     * @param out   resulting dense matrix (n, q).
     */
    virtual void to_dense(
        int j, int q,
        Eigen::Ref<colmat_value_t> out
    ) const =0;

    /**
     * @brief Computes column-wise mean.
     * 
     * @param out   resulting column means.
     */
    virtual void means(
        Eigen::Ref<vec_value_t> out
    ) const =0;

    /**
     * @brief Computes group-wise column norms.
     * 
     * @param groups    mapping group number to starting position.
     * @param group_sizes   mapping group number to group size.
     * @param center        true to compute centered column norms.
     * @param out           resulting group-wise column norms.
     */
    virtual void group_norms(
        const Eigen::Ref<const vec_index_t>& groups,
        const Eigen::Ref<const vec_index_t>& group_sizes,
        const Eigen::Ref<const vec_value_t>& means,
        bool center,
        Eigen::Ref<vec_value_t> out
    ) const =0;

    /**
     * @brief Returns the number of rows of the represented matrix.
     */
    virtual int rows() const =0;
    
    /**
     * @brief Returns the number of columns of the represented matrix.
     */
    virtual int cols() const =0;
};

} // namespace matrix
} // namespace adelie_core