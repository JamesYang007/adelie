#pragma once
#include <adelie_core/state/state_pin_base.hpp>

namespace adelie_core {
namespace state {

template <class MatrixType, 
          class ValueType=typename std::decay_t<MatrixType>::value_t,
          class IndexType=Eigen::Index,
          class BoolType=bool
        >
struct StatePinNaive : StatePinBase<
        ValueType,
        IndexType,
        BoolType
    >
{
    using base_t = StatePinBase<
        ValueType,
        IndexType,
        BoolType
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
    using typename base_t::dyn_vec_mat_value_t;
    using matrix_t = MatrixType;
    using dyn_vec_vec_value_t = std::vector<vec_value_t>;

    /* Static states */
    const value_t y_mean;
    const map_cvec_value_t strong_X_means;

    /* Dynamic states */
    matrix_t* X;
    map_vec_value_t resid;
    value_t resid_sum;
    dyn_vec_vec_value_t resids;

    /* buffer */
    vec_value_t strong_grad;

    explicit StatePinNaive(
        matrix_t& X,
        value_t y_mean,
        const Eigen::Ref<const vec_index_t>& groups, 
        const Eigen::Ref<const vec_index_t>& group_sizes,
        value_t alpha, 
        const Eigen::Ref<const vec_value_t>& penalty,
        const Eigen::Ref<const vec_index_t>& strong_set, 
        const Eigen::Ref<const vec_index_t>& strong_g1,
        const Eigen::Ref<const vec_index_t>& strong_g2,
        const Eigen::Ref<const vec_index_t>& strong_begins, 
        const Eigen::Ref<const vec_value_t>& strong_vars,
        const Eigen::Ref<const vec_value_t>& strong_X_means,
        const dyn_vec_mat_value_t& strong_transforms,
        const Eigen::Ref<const vec_value_t>& lmda_path, 
        bool intercept,
        size_t max_iters,
        value_t tol,
        value_t rsq_slope_tol,
        value_t rsq_curv_tol,
        value_t newton_tol,
        size_t newton_max_iters,
        size_t n_threads,
        value_t rsq,
        Eigen::Ref<vec_value_t> resid,
        value_t resid_sum,
        Eigen::Ref<vec_value_t> strong_beta, 
        Eigen::Ref<vec_bool_t> strong_is_active
    ): 
        base_t(
            groups, group_sizes, alpha, penalty, 
            strong_set, strong_g1, strong_g2, strong_begins, strong_vars, strong_transforms, lmda_path, 
            intercept, max_iters, tol, rsq_slope_tol, rsq_curv_tol, newton_tol, newton_max_iters, n_threads,
            rsq, strong_beta, strong_is_active
        ),
        y_mean(y_mean),
        strong_X_means(strong_X_means.data(), strong_X_means.size()),
        X(&X),
        resid(resid.data(), resid.size()),
        resid_sum(resid_sum),
        strong_grad(strong_beta.size())
    {
        resids.reserve(lmda_path.size());
    }
};

} // namespace state
} // namespace adelie_core