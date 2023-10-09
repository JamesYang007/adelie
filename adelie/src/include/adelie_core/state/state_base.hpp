#pragma once
#include <vector>
#include <unordered_set>
#include <adelie_core/util/types.hpp>

namespace adelie_core {
namespace state {

template <class ValueType,
          class IndexType=Eigen::Index,
          class BoolType=bool,
          class SafeBoolType=unsigned char,
          class USetIndexType=std::unordered_set<IndexType>,
          class DynVecValueType=std::vector<ValueType>,
          class DynVecIndexType=std::vector<IndexType>,
          class DynVecBoolType=std::vector<SafeBoolType>,
          class DynVecSpVecType=std::vector<
            util::sp_vec_type<ValueType, Eigen::RowMajor, IndexType>
          >
        >
struct StateBase
{
    using value_t = ValueType;
    using index_t = IndexType;
    using bool_t = BoolType;
    using safe_bool_t = SafeBoolType;
    using uset_index_t = USetIndexType;
    using vec_value_t = util::rowvec_type<value_t>;
    using vec_index_t = util::rowvec_type<index_t>;
    using sp_vec_value_t = util::sp_vec_type<value_t, Eigen::RowMajor, index_t>;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;
    using map_cvec_index_t = Eigen::Map<const vec_index_t>;
    using dyn_vec_value_t = DynVecValueType;
    using dyn_vec_index_t = DynVecIndexType;
    using dyn_vec_bool_t = DynVecBoolType;
    using dyn_vec_sp_vec_t = DynVecSpVecType;

    /* static states */
    const size_t initial_size;
    const size_t initial_size_groups;

    const map_cvec_index_t groups;
    const map_cvec_index_t group_sizes;
    const value_t alpha;
    const map_cvec_value_t penalty;
    const map_cvec_value_t vars;

    /* configurations */
    // lambda path generation configs
    const value_t min_ratio;
    const size_t n_lambdas;
    const size_t n_lambdas_iter;
    const map_cvec_value_t user_lmdas;

    // basil strong set configs
    const size_t delta_strong_size;
    const size_t max_strong_size;
    const bool strong_rule;

    // convergence configs
    const size_t max_iters;
    const value_t tol;
    const value_t rsq_slope_tol;
    const value_t rsq_curv_tol;
    const value_t newton_tol;
    const size_t newton_max_iters;
    const bool early_exit;

    // other configs
    const bool debug;
    const size_t n_threads;

    /* dynamic states */
    uset_index_t strong_hashset;
    dyn_vec_index_t strong_set; 
    dyn_vec_index_t strong_g1;
    dyn_vec_index_t strong_g2;
    dyn_vec_index_t strong_begins;
    dyn_vec_index_t strong_order;
    dyn_vec_value_t strong_beta;
    dyn_vec_value_t strong_beta_prev_valid;
    dyn_vec_value_t strong_grad;
    dyn_vec_value_t strong_vars;
    dyn_vec_index_t active_set;
    dyn_vec_index_t active_g1;
    dyn_vec_index_t active_g2;
    dyn_vec_index_t active_begins;
    dyn_vec_index_t active_order;
    dyn_vec_bool_t is_active;
    vec_value_t grad;
    vec_value_t grad_next;
    vec_value_t abs_grad;
    vec_value_t abs_grad_next;
    value_t rsq_prev_valid = 0;
    dyn_vec_sp_vec_t betas_curr;
    dyn_vec_value_t rsqs_curr;
    dyn_vec_sp_vec_t betas;
    dyn_vec_value_t rsqs;
    dyn_vec_value_t lmdas;

    /* diagnostics */
    // TODO: fill

    virtual ~StateBase() =default;

    explicit StateBase(
        const Eigen::Ref<const vec_index_t>& groups, 
        const Eigen::Ref<const vec_index_t>& group_sizes,
        value_t alpha, 
        const Eigen::Ref<const vec_value_t>& penalty,
        const Eigen::Ref<const vec_value_t>& vars,
        size_t max_iters,
        value_t tol,
        value_t rsq_slope_tol,
        value_t rsq_curv_tol,
        value_t newton_tol,
        size_t newton_max_iters,
        size_t n_threads,
        value_t rsq,
        Eigen::Ref<vec_value_t> strong_beta, 
        Eigen::Ref<vec_value_t> strong_grad,
        dyn_vec_index_t active_set,
        dyn_vec_index_t active_g1,
        dyn_vec_index_t active_g2,
        dyn_vec_index_t active_begins,
        dyn_vec_index_t active_order,
        Eigen::Ref<vec_bool_t> is_active,
        dyn_vec_sp_vec_t betas, 
        dyn_vec_value_t rsqs
    ): 
        groups(groups.data(), groups.size()),
        group_sizes(group_sizes.data(), group_sizes.size()),
        alpha(alpha),
        penalty(penalty.data(), penalty.size()),
        strong_set(strong_set.data(), strong_set.size()),
        strong_g1(strong_g1.data(), strong_g1.size()),
        strong_g2(strong_g2.data(), strong_g2.size()),
        strong_begins(strong_begins.data(), strong_begins.size()),
        strong_vars(strong_vars.data(), strong_vars.size()),
        lmdas(lmdas.data(), lmdas.size()),
        max_iters(max_iters),
        tol(tol),
        rsq_slope_tol(rsq_slope_tol),
        rsq_curv_tol(rsq_curv_tol),
        newton_tol(newton_tol),
        newton_max_iters(newton_max_iters),
        n_threads(n_threads),
        rsq(rsq),
        strong_beta(strong_beta.data(), strong_beta.size()),
        strong_grad(strong_grad.data(), strong_grad.size()),
        active_set(active_set),
        active_g1(active_g1),
        active_g2(active_g2),
        active_begins(active_begins),
        active_order(active_order),
        is_active(is_active.data(), is_active.size()),
        betas(betas),
        rsqs(rsqs)
    {}
};

} // namespace state
} // namespace adelie_core