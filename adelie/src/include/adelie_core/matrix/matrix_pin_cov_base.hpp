#pragma once
#include <adelie_core/util/types.hpp>

namespace adelie_core {
namespace matrix {

template <class ValueType>
class MatrixPinCovBase
{
public:
    using value_t = ValueType;
    using rowvec_t = util::rowvec_type<value_t>;
    
    virtual ~MatrixPinCovBase() {}

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
     * @brief Returns the ith diagonal element.
     * 
     * @param i     diagonal index.
     */
    virtual value_t diag(int i) const =0;

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
