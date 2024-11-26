#pragma once
#include <adelie_core/util/exceptions.hpp>
#include <adelie_core/util/format.hpp>
#include <adelie_core/util/types.hpp>

#ifndef ADELIE_CORE_MATRIX_NAIVE_BASE_TP
#define ADELIE_CORE_MATRIX_NAIVE_BASE_TP \
    template <class ValueType, class IndexType>
#endif
#ifndef ADELIE_CORE_MATRIX_NAIVE_BASE
#define ADELIE_CORE_MATRIX_NAIVE_BASE \
    MatrixNaiveBase<ValueType, IndexType>
#endif

namespace adelie_core {
namespace matrix {

template <class ValueType, class IndexType=Eigen::Index> 
class MatrixNaiveBase
{
protected:
    static inline void check_cmul(
        int j, int v, int w, int r, int c
    );

    static inline void check_ctmul(
        int j, int o, int r, int c
    );

    static inline void check_bmul(
        int j, int q, int v, int w, int o, int r, int c
    );

    static inline void check_btmul(
        int j, int q, int v, int o, int r, int c
    );

    static inline void check_cov(
        int j, int q, int w, int o_r, int o_c, int r, int c
    );

    static inline void check_sp_tmul(
        int vr, int vc, int o_r, int o_c, int r, int c
    );

public:
    using value_t = ValueType;
    using index_t = IndexType;
    using vec_value_t = util::rowvec_type<value_t>;
    using vec_index_t = util::rowvec_type<index_t>;
    using colmat_value_t = util::colmat_type<value_t>;
    using rowmat_value_t = util::rowmat_type<value_t>;
    using sp_mat_value_t = Eigen::SparseMatrix<value_t, Eigen::RowMajor>;
    
    virtual ~MatrixNaiveBase() {}
    
    virtual value_t cmul(
        int j, 
        const Eigen::Ref<const vec_value_t>& v,
        const Eigen::Ref<const vec_value_t>& weights
    ) =0;

    virtual value_t cmul_safe(
        int j, 
        const Eigen::Ref<const vec_value_t>& v,
        const Eigen::Ref<const vec_value_t>& weights
    ) const =0;

    virtual void ctmul(
        int j, 
        value_t v, 
        Eigen::Ref<vec_value_t> out
    ) =0;

    virtual void bmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) =0;

    virtual void bmul_safe(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) const =0;

    virtual void btmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
    ) =0;

    virtual void mul(
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) const =0;

    virtual void cov(
        int j, int q,
        const Eigen::Ref<const vec_value_t>& sqrt_weights,
        Eigen::Ref<colmat_value_t> out
    ) const =0;

    virtual int rows() const =0;
    
    virtual int cols() const =0;

    /* Non-speed critical routines */

    virtual void mean(
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) const
    {
        vec_value_t ones = vec_value_t::Ones(weights.size());
        mul(ones, weights, out);
    }

    virtual void var(
        const Eigen::Ref<const vec_value_t>& centers,
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) const
    {
        const auto sum_w = weights.sum();
        vec_value_t m(out.size());
        mean(weights, m);
        sq_mul(weights, out);
        out += centers * (centers * sum_w - 2 * m);
    }

    virtual void sq_mul(
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) const =0;

    virtual void sp_tmul(
        const sp_mat_value_t& v,
        Eigen::Ref<rowmat_value_t> out
    ) const =0;
};

ADELIE_CORE_MATRIX_NAIVE_BASE_TP
void
ADELIE_CORE_MATRIX_NAIVE_BASE::check_cmul(
    int j, int v, int w, int r, int c
)
{
    if (
        (j < 0 || j >= c) ||
        (v != r) ||
        (w != r)
    ) {
        throw util::adelie_core_error(
            util::format(
                "cmul() is given inconsistent inputs! "
                "Invoked check_cmul(j=%d, v=%d, w=%d, r=%d, c=%d)",
                j, v, w, r, c
            )
        );
    }
}

ADELIE_CORE_MATRIX_NAIVE_BASE_TP
void
ADELIE_CORE_MATRIX_NAIVE_BASE::check_ctmul(
    int j, int o, int r, int c
)
{
    if (
        (j < 0 || j >= c) ||
        (o != r)
    ) {
        throw util::adelie_core_error(
            util::format(
                "ctmul() is given inconsistent inputs! "
                "Invoked check_ctmul(j=%d, o=%d, r=%d, c=%d)",
                j, o, r, c
            )
        );
    }
}

ADELIE_CORE_MATRIX_NAIVE_BASE_TP
void
ADELIE_CORE_MATRIX_NAIVE_BASE::check_bmul(
    int j, int q, int v, int w, int o, int r, int c
)
{
    if (
        (j < 0 || j > c-q) ||
        (v != r) ||
        (w != r) ||
        (o != q)
    ) {
        throw util::adelie_core_error(
            util::format(
                "bmul() is given inconsistent inputs! "
                "Invoked check_bmul(j=%d, q=%d, v=%d, w=%d, o=%d, r=%d, c=%d)",
                j, q, v, w, o, r, c
            )
        );
    }
}

ADELIE_CORE_MATRIX_NAIVE_BASE_TP
void
ADELIE_CORE_MATRIX_NAIVE_BASE::check_btmul(
    int j, int q, int v, int o, int r, int c
)
{
    if (
        (j < 0 || j > c-q) ||
        (v != q) ||
        (o != r)
    ) {
        throw util::adelie_core_error(
            util::format(
                "btmul() is given inconsistent inputs! "
                "Invoked check_btmul(j=%d, q=%d, v=%d, o=%d, r=%d, c=%d)",
                j, q, v, o, r, c
            )
        );
    }
}

ADELIE_CORE_MATRIX_NAIVE_BASE_TP
void
ADELIE_CORE_MATRIX_NAIVE_BASE::check_cov(
    int j, int q, int w, int o_r, int o_c, int r, int c
)
{
    if (
        (j < 0 || j > c-q) ||
        (w != r) ||
        (o_r != q) ||
        (o_c != q)
    ) {
        throw util::adelie_core_error(
            util::format(
                "cov() is given inconsistent inputs! "
                "Invoked check_cov(j=%d, q=%d, w=%d, o_r=%d, o_c=%d, r=%d, c=%d)",
                j, q, w, o_r, o_c, r, c
            )
        );
    }
}

ADELIE_CORE_MATRIX_NAIVE_BASE_TP
void
ADELIE_CORE_MATRIX_NAIVE_BASE::check_sp_tmul(
    int vr, int vc, int o_r, int o_c, int r, int c
)
{
    if (
        (vr != o_r) || 
        (vc != c) ||
        (o_c != r)
    ) {
        throw util::adelie_core_error(
            util::format(
                "sp_tmul() is given inconsistent inputs! "
                "Invoked check_sp_tmul(vr=%d, vc=%d, o_r=%d, o_c=%d, r=%d, c=%d)",
                vr, vc, o_r, o_c, r, c
            )
        );
    }
}

} // namespace matrix
} // namespace adelie_core

#ifndef ADELIE_CORE_MATRIX_NAIVE_PURE_OVERRIDE_DECL
#define ADELIE_CORE_MATRIX_NAIVE_PURE_OVERRIDE_DECL \
    value_t cmul(\
        int j,\
        const Eigen::Ref<const vec_value_t>& v,\
        const Eigen::Ref<const vec_value_t>& weights\
    ) override;\
    value_t cmul_safe(\
        int j,\
        const Eigen::Ref<const vec_value_t>& v,\
        const Eigen::Ref<const vec_value_t>& weights\
    ) const override;\
    void ctmul(\
        int j,\
        value_t v,\
        Eigen::Ref<vec_value_t> out\
    ) override;\
    void bmul(\
        int j, int q,\
        const Eigen::Ref<const vec_value_t>& v,\
        const Eigen::Ref<const vec_value_t>& weights,\
        Eigen::Ref<vec_value_t> out\
    ) override;\
    void bmul_safe(\
        int j, int q,\
        const Eigen::Ref<const vec_value_t>& v,\
        const Eigen::Ref<const vec_value_t>& weights,\
        Eigen::Ref<vec_value_t> out\
    ) const override;\
    void btmul(\
        int j, int q,\
        const Eigen::Ref<const vec_value_t>& v,\
        Eigen::Ref<vec_value_t> out\
    ) override;\
    void mul(\
        const Eigen::Ref<const vec_value_t>& v,\
        const Eigen::Ref<const vec_value_t>& weights,\
        Eigen::Ref<vec_value_t> out\
    ) const override;\
    void cov(\
        int j, int q,\
        const Eigen::Ref<const vec_value_t>& sqrt_weights,\
        Eigen::Ref<colmat_value_t> out\
    ) const override;\
    int rows() const override;\
    int cols() const override;\
    void sq_mul(\
        const Eigen::Ref<const vec_value_t>& weights,\
        Eigen::Ref<vec_value_t> out\
    ) const override;\
    void sp_tmul(\
        const sp_mat_value_t& v,\
        Eigen::Ref<rowmat_value_t> out\
    ) const override;
#endif

#ifndef ADELIE_CORE_MATRIX_NAIVE_OVERRIDE_DECL
#define ADELIE_CORE_MATRIX_NAIVE_OVERRIDE_DECL \
    void mean(\
        const Eigen::Ref<const vec_value_t>& weights,\
        Eigen::Ref<vec_value_t> out\
    ) const override;\
    void var(\
        const Eigen::Ref<const vec_value_t>& centers,\
        const Eigen::Ref<const vec_value_t>& weights,\
        Eigen::Ref<vec_value_t> out\
    ) const override;
#endif