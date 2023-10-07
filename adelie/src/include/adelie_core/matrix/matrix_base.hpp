#pragma once
#include <adelie_core/util/types.hpp>

namespace adelie_core {
namespace matrix {

template <class ValueType> 
class MatrixNaiveBase
{
public:
    using value_t = ValueType;
    using rowvec_t = util::rowvec_type<value_t>;
    
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
        const Eigen::Ref<const rowvec_t>& v
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
        Eigen::Ref<rowvec_t> out
    ) const =0;

    /**
     * @brief Computes v^T X[i:i+p, j:j+q] where X is the current matrix.
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
        const Eigen::Ref<const rowvec_t>& v, 
        Eigen::Ref<rowvec_t> out
    ) =0;

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
    virtual void btmul(
        int i, int j, int p, int q, 
        const Eigen::Ref<const rowvec_t>& v, 
        Eigen::Ref<rowvec_t> out
    ) =0;

    /**
     * @brief Computes the squared norm of a column of the matrix.
     * 
     * @param j     column index.
     */
    virtual value_t cnormsq(int j) const =0;

    /**
     * @brief Returns the number of rows of the represented matrix.
     */
    virtual int rows() const =0;
    
    /**
     * @brief Returns the number of columns of the represented matrix.
     */
    virtual int cols() const =0;
};

template <class ValueType>
class MatrixCovBase
{
public:
    using value_t = ValueType;
    using rowvec_t = util::rowvec_type<value_t>;
    
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
        const Eigen::Ref<const rowvec_t>& v, 
        Eigen::Ref<rowvec_t> out
    ) =0;

    /**
     * @brief Returns the coefficient at (i, j).
     * 
     * @param i     row index.
     * @param j     column index.
     */
    virtual value_t coeff(int i, int j) const =0;

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