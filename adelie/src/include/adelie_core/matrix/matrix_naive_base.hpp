#pragma once
#include <cstdio>
#include <adelie_core/util/types.hpp>
#include <adelie_core/util/format.hpp>

namespace adelie_core {
namespace matrix {

template <class ValueType, class IndexType=int> 
class MatrixNaiveBase
{
protected:
    static void check_cmul(
        int j, int v, int r, int c
    )
    {
        if (
            (j < 0 || j > c) ||
            (v != r)
        ) {
            throw std::runtime_error(
                util::format(
                    "cmul() is given inconsistent inputs! "
                    "Invoked check_cmul(j=%d, v=%d, r=%d, c=%d)",
                    j, v, r, c
                )
            );
        }
    }

    static void check_ctmul(
        int j, int w, int o, int r, int c
    )
    {
        if (
            (j < 0 || j > c) ||
            (w != r) ||
            (o != r)
        ) {
            throw std::runtime_error(
                util::format(
                    "ctmul() is given inconsistent inputs! "
                    "Invoked check_ctmul(j=%d, w=%d, o=%d, r=%d, c=%d)",
                    j, w, o, r, c
                )
            );
        }
    }

    static void check_bmul(
        int j, int q, int v, int o, int r, int c
    )
    {
        if (
            (j < 0 || j > c-q) ||
            (v != r) ||
            (o != q)
        ) {
            throw std::runtime_error(
                util::format(
                    "bmul() is given inconsistent inputs! "
                    "Invoked check_bmul(j=%d, q=%d, v=%d, o=%d, r=%d, c=%d)",
                    j, q, v, o, r, c
                )
            );
        }
    }

    static void check_btmul(
        int j, int q, int v, int w, int o, int r, int c
    )
    {
        if (
            (j < 0 || j > c-q) ||
            (v != q) ||
            (w != r) ||
            (o != r)
        ) {
            throw std::runtime_error(
                util::format(
                    "btmul() is given inconsistent inputs! "
                    "Invoked check_btmul(j=%d, q=%d, v=%d, w=%d, o=%d, r=%d, c=%d)",
                    j, q, v, w, o, r, c
                )
            );
        }
    }

    static void check_cov(
        int j, int q, int w, int o_r, int o_c, int br, int bc, int r, int c
    )
    {
        if (
            (j < 0 || j > c-q) ||
            (w != r) ||
            (o_r != q) ||
            (o_c != q) ||
            (br != r) ||
            (bc != q)
        ) {
            throw std::runtime_error(
                util::format(
                    "cov() is given inconsistent inputs! "
                    "Invoked check_cov(j=%d, q=%d, w=%d, o_r=%d, o_c=%d, br=%d, bc=%d, r=%d, c=%d)",
                    j, q, w, o_r, o_c, br, bc, r, c
                )
            );
        }
    }

    static void check_sp_btmul(
        int vr, int vc, int w, int o_r, int o_c, int r, int c
    )
    {
        if (
            (vr != o_r) || 
            (vc != c) ||
            (w != r) ||
            (o_c != r)
        ) {
            throw std::runtime_error(
                util::format(
                    "sp_btmul() is given inconsistent inputs! "
                    "Invoked check_sp_btmul(vr=%d, vc=%d, w=%d, o_r=%d, o_c=%d, r=%d, c=%d)",
                    vr, vc, w, o_r, o_c, r, c
                )
            );
        }
    }

    static void check_means(
        int w, int o, int r, int c
    )
    {
        if (
            (w != r) || (o != c)
        ) {
            throw std::runtime_error(
                util::format(
                    "means() is given inconsistent inputs! "
                    "Invoked check_means(w=%d, o=%d, r=%d, c=%d)",
                    w, o, r, c
                )
            );
        }
    }

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
    ) =0;

    /**
     * @brief Computes v X[:, j]^T W where X is the current matrix.
     * 
     * @param j         column index. 
     * @param v         scalar to multiply with. 
     * @param   weights diagonal weights W.
     * @param out       resulting row vector.
     */
    virtual void ctmul(
        int j, 
        value_t v, 
        const Eigen::Ref<const vec_value_t>& weights,
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
     * @brief Computes v^T X[:, j:j+q]^T W where X is the current matrix.
     * 
     * @param j     begin column index. 
     * @param q     number of columns.
     * @param v     vector to multiply with.
     * @param   weights diagonal weights W.
     * @param out   resulting row vector.
     */
    virtual void btmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) =0;

    /**
     * @brief Computes v^T X where X is the current matrix.
     * 
     * @param v     vector to multiply with.
     * @param out   resulting row vector.
     */
    virtual void mul(
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
    ) =0;

    /**
     * @brief Computes weighted covariance matrix.
     * 
     * @param   j       begin column index
     * @param   q       number of columns.
     * @param   sqrt_weights square-root of the weights.
     * @param   means   corresponding column means for columns [j, j+q).
     * @param   center  true if columns should be centered by means.
     * @param   out     resulting covariance matrix.
     * @param   buffer  (n, q) extra buffer space if needed.
     */
    virtual void cov(
        int j, int q,
        const Eigen::Ref<const vec_value_t>& sqrt_weights,
        Eigen::Ref<colmat_value_t> out,
        Eigen::Ref<colmat_value_t> buffer
    ) const =0;

    /**
     * @brief Returns the number of rows of the represented matrix.
     */
    virtual int rows() const =0;
    
    /**
     * @brief Returns the number of columns of the represented matrix.
     */
    virtual int cols() const =0;

    /* Used outside of fitting procedures */

    /**
     * @brief Computes v X^T W where X is the current matrix.
     * 
     * @param v     (l, p) sparse matrix to multiply with.
     * @param   weights diagonal weights W.
     * @param out   (l, n) resulting row vector.
     */
    virtual void sp_btmul(
        const sp_mat_value_t& v,
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<rowmat_value_t> out
    ) const =0;

    /**
     * @brief Computes column-wise mean (weighted by W).
     * 
     * @param   weights diagonal weights W.
     * @param out   resulting column means.
     */
    virtual void means(
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) const =0;
};

} // namespace matrix
} // namespace adelie_core