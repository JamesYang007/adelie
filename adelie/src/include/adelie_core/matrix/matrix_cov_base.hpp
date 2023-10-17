#pragma once
#include <adelie_core/util/types.hpp>

namespace adelie_core {
namespace matrix {

template <class ValueType>
class MatrixCovBase
{
public:
    using value_t = ValueType;
    using vec_value_t = util::rowvec_type<value_t>;
    using colmat_value_t = util::colmat_type<value_t>;
    
    virtual ~MatrixCovBase() {}

    /**
     * @brief Computes v^T X[i:i+p, j:j+q]^T where X is the current matrix.
     * 
     * @param i     begin row index.
     * @param j     begin column index. 
     * @param p     number of rows. 
     * @param q     number of columns.
     * @param v     vector to multiply with.
     * @param out   resulting row vector.
     */
    virtual void bmul(
        int i, int j, int p, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
    ) =0;

    /**
     * @brief Computes the squared norm of a column of the matrix.
     * 
     * @param i     row index.
     * @param j     column index.
     * @param p     number of rows.
     * @param q     number of columns.
     * @param out   resulting dense matrix (n, q).
     */
    virtual void to_dense(
        int i, int j, int p, int q,
        Eigen::Ref<colmat_value_t> out
    ) const =0;

    /**
     * @brief Returns the number of rows of the represented matrix.
     */
    virtual int rows() const { return cols(); };
    
    /**
     * @brief Returns the number of columns of the represented matrix.
     */
    virtual int cols() const =0;
};

} // namespace matrix
} // namespace adelie_core
