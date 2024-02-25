#pragma once
#include <adelie_core/util/types.hpp>
#include <adelie_core/util/format.hpp>

namespace adelie_core {
namespace matrix {

template <class ValueType>
class MatrixCovBase
{
protected:
    static void check_bmul(
        int i, int j, int p, int q, int v, int o, int r, int c
    )
    {
        if (
            (i < 0 || i > r-p) ||
            (j < 0 || j > c-q) ||
            (v != p) ||
            (o != q)
        ) {
            throw std::runtime_error(
                util::format(
                    "bmul() is given inconsistent inputs! "
                    "Invoked check_bmul(i=%d, j=%d, p=%d, q=%d, v=%d, o=%d, r=%d, c=%d)",
                    i, j, p, q, v, o, r, c
                )
            );
        }
    }

    static void check_mul(
        int i, int p, int v, int o, int r, int c
    )
    {
        if (
            (i < 0 || i > r-p) ||
            (v != p) ||
            (o != c) ||
            (r != c)
        ) {
            throw std::runtime_error(
                util::format(
                    "bmul() is given inconsistent inputs! "
                    "Invoked check_bmul(i=%d, p=%d, v=%d, o=%d, r=%d, c=%d)",
                    i, p, v, o, r, c
                )
            );
        }
    }

    static void check_to_dense(
        int i, int p, int o_r, int o_c, int r, int c
    )
    {
        if (
            (i < 0 || i > r-p) ||
            (o_r != p) ||
            (o_c != p) ||
            (r != c)
        ) {
            throw std::runtime_error(
                util::format(
                    "to_dense() is given inconsistent inputs! "
                    "Invoked check_to_dense(i=%d, p=%d, o_r=%d, o_c=%d, r=%d, c=%d)",
                    i, p, o_r, o_c, r, c
                )
            );
        }
    }

public:
    using value_t = ValueType;
    using vec_value_t = util::rowvec_type<value_t>;
    using colmat_value_t = util::colmat_type<value_t>;
    
    virtual ~MatrixCovBase() {}

    virtual void bmul(
        int i, int j, int p, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
    ) =0;

    virtual void mul(
        int i, int p,
        const Eigen::Ref<const vec_value_t>& v,
        Eigen::Ref<vec_value_t> out
    ) =0;

    virtual void to_dense(
        int i, int p, 
        Eigen::Ref<colmat_value_t> out
    ) =0;

    int rows() const { return cols(); }
    
    virtual int cols() const =0;
};

} // namespace matrix
} // namespace adelie_core
