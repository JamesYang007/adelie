#pragma once
#include <adelie_core/util/exceptions.hpp>
#include <adelie_core/util/format.hpp>
#include <adelie_core/util/types.hpp>

#ifndef ADELIE_CORE_MATRIX_COV_BASE_TP
#define ADELIE_CORE_MATRIX_COV_BASE_TP \
    template <class ValueType, class IndexType>
#endif
#ifndef ADELIE_CORE_MATRIX_COV_BASE
#define ADELIE_CORE_MATRIX_COV_BASE \
    MatrixCovBase<ValueType, IndexType>
#endif

namespace adelie_core {
namespace matrix {

template <class ValueType, class IndexType=Eigen::Index>
class MatrixCovBase
{
protected:
    static inline void check_bmul(
        int s, int i, int v, int o, int r, int c
    );

    static inline void check_mul(
        int i, int v, int o, int r, int c
    );

    static inline void check_to_dense(
        int i, int p, int o_r, int o_c, int r, int c
    );

public:
    using value_t = ValueType;
    using index_t = IndexType;
    using vec_index_t = util::rowvec_type<index_t>;
    using vec_value_t = util::rowvec_type<value_t>;
    using colmat_value_t = util::colmat_type<value_t>;
    
    virtual ~MatrixCovBase() {}

    virtual void bmul(
        const Eigen::Ref<const vec_index_t>& subset,
        const Eigen::Ref<const vec_index_t>& indices,
        const Eigen::Ref<const vec_value_t>& values,
        Eigen::Ref<vec_value_t> out
    ) =0;

    virtual void mul(
        const Eigen::Ref<const vec_index_t>& indices,
        const Eigen::Ref<const vec_value_t>& values,
        Eigen::Ref<vec_value_t> out
    ) const =0;

    virtual void to_dense(
        int i, int p, 
        Eigen::Ref<colmat_value_t> out
    ) const =0;

    int rows() const { return cols(); }
    
    virtual int cols() const =0;
};

ADELIE_CORE_MATRIX_COV_BASE_TP
void
ADELIE_CORE_MATRIX_COV_BASE::check_bmul(
    int s, int i, int v, int o, int r, int c
)
{
    if (
        (s < 0 || s > r) ||
        (i < 0 || i > r) ||
        (i != v) ||
        (v < 0 || v > r) ||
        (o != s)
    ) {
        throw util::adelie_core_error(
            util::format(
                "bmul() is given inconsistent inputs! "
                "Invoked check_bmul(s=%d, i=%d, v=%d, o=%d, r=%d, c=%d)",
                s, i, v, o, r, c
            )
        );
    }
}

ADELIE_CORE_MATRIX_COV_BASE_TP
void
ADELIE_CORE_MATRIX_COV_BASE::check_mul(
    int i, int v, int o, int r, int c
)
{
    if (
        (i < 0 || i > r) ||
        (i != v) ||
        (o != c) ||
        (r != c)
    ) {
        throw util::adelie_core_error(
            util::format(
                "mul() is given inconsistent inputs! "
                "Invoked check_mul(i=%d, v=%d, o=%d, r=%d, c=%d)",
                i, v, o, r, c
            )
        );
    }
}

ADELIE_CORE_MATRIX_COV_BASE_TP
void
ADELIE_CORE_MATRIX_COV_BASE::check_to_dense(
    int i, int p, int o_r, int o_c, int r, int c
)
{
    if (
        (i < 0 || i > r-p) ||
        (o_r != p) ||
        (o_c != p) ||
        (r != c)
    ) {
        throw util::adelie_core_error(
            util::format(
                "to_dense() is given inconsistent inputs! "
                "Invoked check_to_dense(i=%d, p=%d, o_r=%d, o_c=%d, r=%d, c=%d)",
                i, p, o_r, o_c, r, c
            )
        );
    }
}

} // namespace matrix
} // namespace adelie_core

#ifndef ADELIE_CORE_MATRIX_COV_PURE_OVERRIDE_DECL
#define ADELIE_CORE_MATRIX_COV_PURE_OVERRIDE_DECL \
    void bmul(\
        const Eigen::Ref<const vec_index_t>& subset,\
        const Eigen::Ref<const vec_index_t>& indices,\
        const Eigen::Ref<const vec_value_t>& values,\
        Eigen::Ref<vec_value_t> out\
    ) override;\
    void mul(\
        const Eigen::Ref<const vec_index_t>& indices,\
        const Eigen::Ref<const vec_value_t>& values,\
        Eigen::Ref<vec_value_t> out\
    ) const override;\
    void to_dense(\
        int i, int p,\
        Eigen::Ref<colmat_value_t> out\
    ) const override;\
    int cols() const override;
#endif