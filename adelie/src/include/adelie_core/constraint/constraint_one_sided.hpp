#pragma once
#include <adelie_core/constraint/constraint_base.hpp>

#ifndef ADELIE_CORE_CONSTRAINT_ONE_SIDED_TP
#define ADELIE_CORE_CONSTRAINT_ONE_SIDED_TP \
    template <class ValueType, class IndexType>
#endif
#ifndef ADELIE_CORE_CONSTRAINT_ONE_SIDED
#define ADELIE_CORE_CONSTRAINT_ONE_SIDED \
    ConstraintOneSided<ValueType, IndexType>
#endif

#ifndef ADELIE_CORE_CONSTRAINT_ONE_SIDED_ADMM_TP
#define ADELIE_CORE_CONSTRAINT_ONE_SIDED_ADMM_TP \
    template <class ValueType>
#endif
#ifndef ADELIE_CORE_CONSTRAINT_ONE_SIDED_ADMM
#define ADELIE_CORE_CONSTRAINT_ONE_SIDED_ADMM \
    ConstraintOneSidedADMM<ValueType>
#endif

namespace adelie_core {
namespace constraint {

template <class ValueType, class IndexType=Eigen::Index>
class ConstraintOneSided: public ConstraintBase<ValueType, IndexType>
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
    const map_cvec_value_t _sgn;
    const map_cvec_value_t _b;

    const size_t _max_iters;
    const value_t _tol;
    const size_t _pinball_max_iters;
    const value_t _pinball_tol;
    const value_t _slack;

    vec_value_t _mu;

public:
    explicit ConstraintOneSided(
        const Eigen::Ref<const vec_value_t>& sgn,
        const Eigen::Ref<const vec_value_t>& b,
        size_t max_iters,
        value_t tol,
        size_t pinball_max_iters,
        value_t pinball_tol,
        value_t slack
    );

    ADELIE_CORE_CONSTRAINT_PURE_OVERRIDE_DECL
};

template <class ValueType>
class ConstraintOneSidedADMM: public ConstraintBase<ValueType>
{
public:
    using base_t = ConstraintBase<ValueType>;
    using typename base_t::value_t;
    using typename base_t::vec_index_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_uint64_t;
    using typename base_t::colmat_value_t;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;

private:
    const map_cvec_value_t _sgn;
    const map_cvec_value_t _b;

    const size_t _max_iters;
    const value_t _tol_abs;
    const value_t _tol_rel;
    const value_t _rho;

    vec_value_t _mu;

public:
    explicit ConstraintOneSidedADMM(
        const Eigen::Ref<const vec_value_t>& sgn,
        const Eigen::Ref<const vec_value_t>& b,
        size_t max_iters,
        value_t tol_abs,
        value_t tol_rel,
        value_t rho
    );

    ADELIE_CORE_CONSTRAINT_PURE_OVERRIDE_DECL
};

} // namespace constraint
} // namespace adelie_core