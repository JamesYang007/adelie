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
    static void check_update_coordinate(
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
                    "update_coordinate() is given inconsistent inputs! "
                    "Invoked update_coordinate(x=%d, mu=%d, q=%d, l=%d, m=%d, d=%d)",
                    x, mu, q, l, m, d
                )
            );
        }
    }

    static void check_update_lagrangian(
        int x, int mu, int o, int m, int d
    ) {
        if (
            (x != o) ||
            (o != d) ||
            (mu != m)
        ) {
            throw util::adelie_core_error(
                util::format(
                    "update_lagrangian() is given inconsistent inputs! "
                    "Invoked update_lagrangian(x=%d, mu=%d, o=%d, m=%d, d=%d)",
                    x, mu, o, m, d
                )
            );
        }
    }

public:
    using value_t = ValueType;
    using vec_value_t = util::rowvec_type<value_t>;

    virtual ~ConstraintBase() {}

    virtual void update_coordinate(
        Eigen::Ref<vec_value_t> x,
        Eigen::Ref<vec_value_t> mu,
        const Eigen::Ref<const vec_value_t>& quad,
        const Eigen::Ref<const vec_value_t>& linear,
        value_t l1,
        value_t l2
    ) =0;

    virtual void update_lagrangian(
        const Eigen::Ref<const vec_value_t>& x,
        const Eigen::Ref<const vec_value_t>& mu,
        Eigen::Ref<vec_value_t> out
    ) =0;

    virtual int dual_size() =0;
};

} // namespace constraint
} // namespace adelie_core