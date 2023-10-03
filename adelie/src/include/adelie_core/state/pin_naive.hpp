#pragma once
#include <adelie_core/util/types.hpp>

namespace adelie_core {
namespace state {

/**
 * State class for solve_pin_naive method.
 */
template <class MatrixType, 
          class ValueType=typename std::decay_t<MatrixType>::value_t,
          class IndexType=Eigen::Index,
          class BoolType=bool,
          class DynamicVectorIndexType=std::vector<IndexType>,
          class DynamicVectorValueType=std::vector<ValueType>,
          class DynamicVectorVecValueType=std::vector<util::rowvec_type<ValueType>>,
          class DynamicVectorSpVecType=std::vector<
                util::sp_vec_type<ValueType, Eigen::RowMajor, IndexType>
            > 
          >
struct PinNaive
{
    using value_t = ValueType;
    using index_t = IndexType;
    using bool_t = BoolType;
    using sp_vec_value_t = util::sp_vec_type<value_t, Eigen::RowMajor, index_t>;
    using vec_value_t = util::rowvec_type<value_t>;
    using vec_index_t = util::rowvec_type<index_t>;
    using vec_bool_t = util::rowvec_type<bool_t>;
    using map_vec_value_t = Eigen::Map<vec_value_t>;
    using map_vec_index_t = Eigen::Map<vec_index_t>;
    using map_vec_bool_t = Eigen::Map<vec_bool_t>;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;
    using map_cvec_index_t = Eigen::Map<const vec_index_t>;
    using dyn_vec_index_t = DynamicVectorIndexType;
    using dyn_vec_value_t = DynamicVectorValueType;
    using dyn_vec_vec_value_t = DynamicVectorVecValueType;
    using dyn_vec_sp_vec_t = DynamicVectorSpVecType;

    /* Static states */
    const MatrixType* X;
    const map_cvec_index_t groups;
    const map_cvec_index_t group_sizes;
    const value_t alpha;
    const map_cvec_value_t penalty;
    const map_cvec_index_t strong_set;
    const map_cvec_index_t strong_g1;
    const map_cvec_index_t strong_g2;
    const map_cvec_index_t strong_begins;
    const map_cvec_value_t strong_var;
    const map_cvec_value_t lmdas;

    /* Configurations */
    const size_t max_iters;
    const value_t tol;
    const value_t rsq_slope_tol;
    const value_t rsq_curv_tol;
    const value_t newton_tol;
    const size_t newton_max_iters;

    /* Dynamic states */
    value_t rsq;
    map_vec_value_t resid;
    map_vec_value_t strong_beta;
    map_vec_value_t strong_grad;
    dyn_vec_index_t active_set;
    dyn_vec_index_t active_g1;
    dyn_vec_index_t active_g2;
    dyn_vec_index_t active_begins;
    dyn_vec_index_t active_order;
    map_vec_bool_t is_active;
    dyn_vec_sp_vec_t betas;
    dyn_vec_value_t rsqs;
    dyn_vec_vec_value_t resids;
    size_t n_cds = 0;

    // Benchmark information
    std::vector<double> time_strong_cd;
    std::vector<double> time_active_cd;
    
    explicit PinNaive(
        const MatrixType& X,
        const Eigen::Ref<const vec_index_t>& groups, 
        const Eigen::Ref<const vec_index_t>& group_sizes,
        value_t alpha, 
        const Eigen::Ref<const vec_value_t>& penalty,
        const Eigen::Ref<const vec_index_t>& strong_set, 
        const Eigen::Ref<const vec_index_t>& strong_g1,
        const Eigen::Ref<const vec_index_t>& strong_g2,
        const Eigen::Ref<const vec_index_t>& strong_begins, 
        const Eigen::Ref<const vec_value_t>& strong_var,
        const Eigen::Ref<const vec_value_t>& lmdas, 
        size_t max_iters,
        value_t tol,
        value_t rsq_slope_tol,
        value_t rsq_curv_tol,
        value_t newton_tol,
        size_t newton_max_iters,
        value_t rsq,
        Eigen::Ref<vec_value_t> resid,
        Eigen::Ref<vec_value_t> strong_beta, 
        Eigen::Ref<vec_value_t> strong_grad,
        dyn_vec_index_t active_set,
        dyn_vec_index_t active_g1,
        dyn_vec_index_t active_g2,
        dyn_vec_index_t active_begins,
        dyn_vec_index_t active_order,
        Eigen::Ref<vec_bool_t> is_active,
        dyn_vec_sp_vec_t betas, 
        dyn_vec_value_t rsqs,
        dyn_vec_vec_value_t resids
    )
        : X(&X),
          groups(groups.data(), groups.size()),
          group_sizes(group_sizes.data(), group_sizes.size()),
          alpha(alpha),
          penalty(penalty.data(), penalty.size()),
          strong_set(strong_set.data(), strong_set.size()),
          strong_g1(strong_g1.data(), strong_g1.size()),
          strong_g2(strong_g2.data(), strong_g2.size()),
          strong_begins(strong_begins.data(), strong_begins.size()),
          strong_var(strong_var.data(), strong_var.size()),
          lmdas(lmdas.data(), lmdas.size()),
          max_iters(max_iters),
          tol(tol),
          rsq_slope_tol(rsq_slope_tol),
          rsq_curv_tol(rsq_curv_tol),
          newton_tol(newton_tol),
          newton_max_iters(newton_max_iters),
          rsq(rsq),
          resid(resid.data(), resid.size()),
          strong_beta(strong_beta.data(), strong_beta.size()),
          strong_grad(strong_grad.data(), strong_grad.size()),
          active_set(active_set),
          active_g1(active_g1),
          active_g2(active_g2),
          active_begins(active_begins),
          active_order(active_order),
          is_active(is_active.data(), is_active.size()),
          betas(betas),
          rsqs(rsqs),
          resids(resids)
    {}
};

} // namespace state
} // namespace adelie_core