#pragma once
#include <adelie_core/util/exceptions.hpp>
#include <adelie_core/util/format.hpp>
#include <adelie_core/util/types.hpp>

#ifndef ADELIE_CORE_CONSTRAINT_BASE_TP
#define ADELIE_CORE_CONSTRAINT_BASE_TP \
    template <class ValueType, class IndexType>
#endif
#ifndef ADELIE_CORE_CONSTRAINT_BASE
#define ADELIE_CORE_CONSTRAINT_BASE \
    ConstraintBase<ValueType, IndexType>
#endif

namespace adelie_core {
namespace constraint {

template <class ValueType, class IndexType=Eigen::Index>
class ConstraintBase
{
public:
    using value_t = ValueType;
    using index_t = IndexType;
    using vec_index_t = util::rowvec_type<index_t>;
    using vec_value_t = util::rowvec_type<value_t>;
    using vec_uint64_t = util::rowvec_type<uint64_t>;
    using colmat_value_t = util::colmat_type<value_t>;

protected:
    static inline void check_solve(
        int x, int q, int l, int m, int d
    );

    static inline void check_gradient(
        int x, int mu, int o, int m, int d
    );

public:
    virtual ~ConstraintBase() {}

    virtual void solve(
        Eigen::Ref<vec_value_t> x,
        const Eigen::Ref<const vec_value_t>& quad,
        const Eigen::Ref<const vec_value_t>& linear,
        value_t l1,
        value_t l2,
        const Eigen::Ref<const colmat_value_t>& Q,
        Eigen::Ref<vec_uint64_t> buffer
    ) =0;

    virtual void gradient(
        const Eigen::Ref<const vec_value_t>& x,
        Eigen::Ref<vec_value_t> out
    ) const =0;

    // TODO: gradient(x, mu_indices, mu_values, out)

    virtual void gradient(
        const Eigen::Ref<const vec_value_t>& x,
        const Eigen::Ref<const vec_value_t>& mu,
        Eigen::Ref<vec_value_t> out
    ) const =0;

    virtual void project(
        Eigen::Ref<vec_value_t> x
    ) const =0;

    virtual value_t solve_zero(
        const Eigen::Ref<const vec_value_t>& v,
        Eigen::Ref<vec_uint64_t> buff
    ) =0;

    virtual void clear() =0;

    virtual void dual(
        Eigen::Ref<vec_index_t> indices,
        Eigen::Ref<vec_value_t> values
    ) const =0;
    virtual int duals_nnz() const =0;
    virtual int duals() const =0;
    virtual int primals() const =0;
    virtual size_t buffer_size() const =0;
};

ADELIE_CORE_CONSTRAINT_BASE_TP
void
ADELIE_CORE_CONSTRAINT_BASE::check_solve(
    int x, int q, int l, int m, int d
) {
    if (
        (x != q) ||
        (q != l) ||
        (l != d)
    ) {
        throw util::adelie_core_error(
            util::format(
                "solve() is given inconsistent inputs! "
                "Invoked solve(x=%d, q=%d, l=%d, m=%d, d=%d)",
                x, q, l, m, d
            )
        );
    }
}

ADELIE_CORE_CONSTRAINT_BASE_TP
void
ADELIE_CORE_CONSTRAINT_BASE::check_gradient(
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

} // namespace constraint
} // namespace adelie_core

#ifndef ADELIE_CORE_CONSTRAINT_PURE_OVERRIDE_DECL
#define ADELIE_CORE_CONSTRAINT_PURE_OVERRIDE_DECL \
    void solve(\
        Eigen::Ref<vec_value_t> x,\
        const Eigen::Ref<const vec_value_t>& quad,\
        const Eigen::Ref<const vec_value_t>& linear,\
        value_t l1,\
        value_t l2,\
        const Eigen::Ref<const colmat_value_t>& Q,\
        Eigen::Ref<vec_uint64_t> buffer\
    ) override;\
    void gradient(\
        const Eigen::Ref<const vec_value_t>& x,\
        Eigen::Ref<vec_value_t> out\
    ) const override;\
    void gradient(\
        const Eigen::Ref<const vec_value_t>& x,\
        const Eigen::Ref<const vec_value_t>& mu,\
        Eigen::Ref<vec_value_t> out\
    ) const override;\
    void project(\
        Eigen::Ref<vec_value_t> x\
    ) const override;\
    value_t solve_zero(\
        const Eigen::Ref<const vec_value_t>& v,\
        Eigen::Ref<vec_uint64_t> buff\
    ) override;\
    void clear() override;\
    void dual(\
        Eigen::Ref<vec_index_t> indices,\
        Eigen::Ref<vec_value_t> values\
    ) const override;\
    int duals_nnz() const override;\
    int duals() const override;\
    int primals() const override;\
    size_t buffer_size() const override;
#endif