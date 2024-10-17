#pragma once
#include <adelie_core/constraint/constraint_base.hpp>

#ifndef ADELIE_CORE_CONSTRAINT_BOX_TP
#define ADELIE_CORE_CONSTRAINT_BOX_TP \
    template <class ValueType, class IndexType>
#endif
#ifndef ADELIE_CORE_CONSTRAINT_BOX
#define ADELIE_CORE_CONSTRAINT_BOX \
    ConstraintBox<ValueType, IndexType>
#endif

namespace adelie_core {
namespace constraint {

template <class ValueType, class IndexType=Eigen::Index>
class ConstraintBox: public ConstraintBase<ValueType, IndexType>
{
public:
    using base_t = ConstraintBase<ValueType, IndexType>;
    using typename base_t::value_t;
    using typename base_t::vec_index_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_uint64_t;
    using typename base_t::colmat_value_t;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;

private:
    const map_cvec_value_t _l;
    const map_cvec_value_t _u;

    const size_t _max_iters;
    const value_t _tol;
    const size_t _pinball_max_iters;
    const value_t _pinball_tol;
    const value_t _slack;

    vec_value_t _mu;

    inline void solve_1d(
        Eigen::Ref<vec_value_t> x,
        Eigen::Ref<vec_value_t> mu,
        const Eigen::Ref<const vec_value_t>& quad,
        const Eigen::Ref<const vec_value_t>& linear,
        value_t l1,
        value_t l2,
        const Eigen::Ref<const colmat_value_t>& Q
    ) const;

public:
    explicit ConstraintBox(
        const Eigen::Ref<const vec_value_t>& l,
        const Eigen::Ref<const vec_value_t>& u,
        size_t max_iters,
        value_t tol,
        size_t pinball_max_iters,
        value_t pinball_tol,
        value_t slack
    );

    ADELIE_CORE_CONSTRAINT_PURE_OVERRIDE_DECL
};

} // namespace constraint
} // namespace adelie_core