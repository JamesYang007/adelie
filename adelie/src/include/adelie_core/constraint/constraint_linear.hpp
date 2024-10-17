#pragma once
#include <unordered_set>
#include <vector>
#include <adelie_core/constraint/constraint_base.hpp>

#ifndef ADELIE_CORE_CONSTRAINT_LINEAR_TP
#define ADELIE_CORE_CONSTRAINT_LINEAR_TP \
    template <class AType, class IndexType>
#endif
#ifndef ADELIE_CORE_CONSTRAINT_LINEAR
#define ADELIE_CORE_CONSTRAINT_LINEAR \
    ConstraintLinear<AType, IndexType>
#endif

namespace adelie_core {
namespace constraint { 

template <class AType, class IndexType=Eigen::Index>
class ConstraintLinear: public ConstraintBase<
    typename std::decay_t<AType>::value_t, 
    IndexType
>
{
public:
    using A_t = std::decay_t<AType>;
    using base_t = ConstraintBase<typename A_t::value_t, IndexType>;
    using typename base_t::index_t;
    using typename base_t::value_t;
    using typename base_t::vec_index_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_uint64_t;
    using typename base_t::colmat_value_t;
    using bool_t = bool;
    using rowmat_value_t = util::rowmat_type<value_t>;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;
    using map_crowmat_value_t = Eigen::Map<const rowmat_value_t>;
    using map_ccolmat_value_t = Eigen::Map<const colmat_value_t>;

private:
    static constexpr value_t _eps = 1e-16; // ignorable coefficient magnitude

    using base_t::check_solve;

    A_t* _A;
    const map_cvec_value_t _l;
    const map_cvec_value_t _u;
    const map_cvec_value_t _A_vars;
    const size_t _max_iters;
    const value_t _tol;
    const size_t _nnls_max_iters;
    const value_t _nnls_tol;
    const size_t _pinball_max_iters;
    const value_t _pinball_tol;
    const value_t _slack;
    const size_t _n_threads;

    std::unordered_set<index_t> _mu_active_set;
    std::unordered_set<index_t> _mu_active_set_prev;
    std::vector<index_t> _mu_active;
    std::vector<index_t> _mu_active_prev;
    std::vector<value_t> _mu_value;
    std::vector<value_t> _mu_value_prev;
    vec_value_t _ATmu;

    inline void compute_ATmu(
        Eigen::Ref<vec_value_t> out
    ); 

    inline void mu_to_dense(
        Eigen::Ref<vec_value_t> mu
    );

    inline void mu_to_sparse(
        Eigen::Ref<vec_value_t> mu
    );

    inline void mu_prune(value_t eps=0);

    inline void _clear();

public:
    explicit ConstraintLinear(
        A_t& A,
        const Eigen::Ref<const vec_value_t>& l,
        const Eigen::Ref<const vec_value_t>& u,
        const Eigen::Ref<const vec_value_t>& A_vars,
        size_t max_iters,
        value_t tol,
        size_t nnls_max_iters,
        value_t nnls_tol,
        size_t pinball_max_iters,
        value_t pinball_tol,
        value_t slack,
        size_t n_threads
    );

    ADELIE_CORE_CONSTRAINT_PURE_OVERRIDE_DECL
};

} // namespace constraint
} // namespace adelie_core