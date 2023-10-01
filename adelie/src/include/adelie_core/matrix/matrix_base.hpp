#pragma once
#include <adelie_core/util/types.hpp>

namespace adelie_core {
namespace matrix {

/**
 * @brief Base class for all matrix classes for fitting group elastic net.
 * 
 * @tparam ValueType  underlying value type.
 */
template <class ValueType> 
class MatrixBase
{
public:
    using value_t = ValueType;
    using vec_t = Eigen::Matrix<value_t, Eigen::Dynamic, 1>;
    using mat_t = Eigen::Matrix<value_t, Eigen::Dynamic, Eigen::Dynamic>;
    
    virtual ~MatrixBase() {}
    
    /**
     * @brief Creates a view of a block of the matrix.
     * 
     * @param i     starting row index.
     * @param j     starting column index.
     * @param p     number of rows.
     * @param q     number of columns.
     */
    virtual Eigen::Ref<const mat_t> block(int i, int j, int p, int q) const =0;

    /**
     * @brief Creates a view of a column of the matrix.
     * 
     * @param j     column index.
     */
    virtual Eigen::Ref<const vec_t> col(int j) const =0;

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