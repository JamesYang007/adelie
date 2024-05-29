#pragma once
#include <cstdio>
#include <adelie_core/util/exceptions.hpp>
#include <adelie_core/util/format.hpp>
#include <adelie_core/util/types.hpp>

namespace adelie_core {
namespace constraint {

template <class ValueType>
class ConstraintBase
{
protected:
    static void check_solve(
        int x, int mu, int q, int l, int m, int d
    ) {
        if (
            (x != q) ||
            (q != l) ||
            (l != d) ||
            (mu != m)
        ) {
            throw util::adelie_core_error(
                util::format(
                    "solve() is given inconsistent inputs! "
                    "Invoked solve(x=%d, mu=%d, q=%d, l=%d, m=%d, d=%d)",
                    x, mu, q, l, m, d
                )
            );
        }
    }

    static void check_gradient(
        int x, int mu, int o, int m, int d
    ) {
        if (
            (x != o) ||
            (o != d) ||
            (mu != m)
        ) {
            throw util::adelie_core_error(
                util::format(
                    "gradient() is given inconsistent inputs! "
                    "Invoked gradient(x=%d, mu=%d, o=%d, m=%d, d=%d)",
                    x, mu, o, m, d
                )
            );
        }
    }

public:
    using value_t = ValueType;
    using vec_value_t = util::rowvec_type<value_t>;
    using colmat_value_t = util::colmat_type<value_t>;

    virtual ~ConstraintBase() {}

    virtual void solve(
        Eigen::Ref<vec_value_t> x,
        Eigen::Ref<vec_value_t> mu,
        const Eigen::Ref<const vec_value_t>& quad,
        const Eigen::Ref<const vec_value_t>& linear,
        value_t l1,
        value_t l2,
        const Eigen::Ref<const colmat_value_t>& Q
    ) =0;

    virtual void gradient(
        const Eigen::Ref<const vec_value_t>& x,
        const Eigen::Ref<const vec_value_t>& mu,
        Eigen::Ref<vec_value_t> out
    ) =0;

    virtual void project(
        Eigen::Ref<vec_value_t> x
    )
    {}

    virtual int duals() =0;
    virtual int primals() =0;
};

} // namespace constraint
} // namespace adelie_core