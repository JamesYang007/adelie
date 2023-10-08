#pragma once
#include <adelie_core/state/state_pin_base.hpp>

namespace adelie_core {
namespace state {

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
struct StatePinNaive : StatePinBase<
        ValueType,
        IndexType,
        BoolType,
        DynamicVectorIndexType,
        DynamicVectorValueType,
        DynamicVectorSpVecType
    >
{
    using base_t = StatePinBase<
        ValueType,
        IndexType,
        BoolType,
        DynamicVectorIndexType,
        DynamicVectorValueType,
        DynamicVectorSpVecType
    >;
    using typename base_t::value_t;
    using typename base_t::index_t;
    using typename base_t::bool_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::vec_bool_t;
    using typename base_t::sp_vec_value_t;
    using typename base_t::map_vec_value_t;
    using typename base_t::map_vec_bool_t;
    using typename base_t::map_cvec_value_t;
    using typename base_t::map_cvec_index_t;
    using typename base_t::dyn_vec_index_t;
    using typename base_t::dyn_vec_value_t;
    using typename base_t::dyn_vec_sp_vec_t;
    using matrix_t = MatrixType;
    using dyn_vec_vec_value_t = DynamicVectorVecValueType;

    /* Static states */

    /* Dynamic states */
    matrix_t* X;
    map_vec_value_t resid;
    dyn_vec_vec_value_t resids;

    explicit StatePinNaive(
        matrix_t& X,
        const Eigen::Ref<const vec_index_t>& groups, 
        const Eigen::Ref<const vec_index_t>& group_sizes,
        value_t alpha, 
        const Eigen::Ref<const vec_value_t>& penalty,
        const Eigen::Ref<const vec_index_t>& strong_set, 
        const Eigen::Ref<const vec_index_t>& strong_g1,
        const Eigen::Ref<const vec_index_t>& strong_g2,
        const Eigen::Ref<const vec_index_t>& strong_begins, 
        const Eigen::Ref<const vec_value_t>& strong_vars,
        const Eigen::Ref<const vec_value_t>& lmdas, 
        size_t max_iters,
        value_t tol,
        value_t rsq_slope_tol,
        value_t rsq_curv_tol,
        value_t newton_tol,
        size_t newton_max_iters,
        size_t n_threads,
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
    ): 
        base_t(
            groups, group_sizes, alpha, penalty, 
            strong_set, strong_g1, strong_g2, strong_begins, strong_vars, lmdas, 
            max_iters, tol, rsq_slope_tol, rsq_curv_tol, newton_tol, newton_max_iters, n_threads,
            rsq, strong_beta, strong_grad, 
            active_set, active_g1, active_g2, active_begins, active_order, is_active,
            betas, rsqs
        ),
        X(&X),
        resid(resid.data(), resid.size()),
        resids(resids)
    {}
};

} // namespace state
} // namespace adelie_core