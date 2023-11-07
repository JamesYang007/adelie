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
    const map_cvec_value_t weights;
    const value_t y_mean;
    const value_t y_var;
    const map_cvec_value_t screen_X_means;

    const value_t rsq_tol;

    /* Dynamic states */
    matrix_t* X;
    map_vec_value_t resid;
    value_t resid_sum;
    dyn_vec_vec_value_t resids;
    dyn_vec_value_t resid_sums;

    /* buffer */
    vec_value_t screen_grad;

    explicit StatePinNaive(
        matrix_t& X,
        value_t y_mean,
        value_t y_var,
        const Eigen::Ref<const vec_index_t>& groups, 
        const Eigen::Ref<const vec_index_t>& group_sizes,
        value_t alpha, 
        const Eigen::Ref<const vec_value_t>& penalty,
        const Eigen::Ref<const vec_value_t>& weights,
        const Eigen::Ref<const vec_index_t>& screen_set, 
        const Eigen::Ref<const vec_index_t>& screen_g1,
        const Eigen::Ref<const vec_index_t>& screen_g2,
        const Eigen::Ref<const vec_index_t>& screen_begins, 
        const Eigen::Ref<const vec_value_t>& screen_vars,
        const Eigen::Ref<const vec_value_t>& screen_X_means,
        const dyn_vec_mat_value_t& screen_transforms,
        const Eigen::Ref<const vec_value_t>& lmda_path, 
        bool intercept,
        size_t max_iters,
        value_t tol,
        value_t rsq_tol,
        value_t rsq_slope_tol,
        value_t rsq_curv_tol,
        value_t newton_tol,
        size_t newton_max_iters,
        size_t n_threads,
        value_t rsq,
        Eigen::Ref<vec_value_t> resid,
        value_t resid_sum,
        Eigen::Ref<vec_value_t> screen_beta, 
        Eigen::Ref<vec_bool_t> screen_is_active
    ): 
        base_t(
            groups, group_sizes, alpha, penalty,
            screen_set, screen_g1, screen_g2, screen_begins, screen_vars, screen_transforms, lmda_path, 
            intercept, max_iters, tol, rsq_slope_tol, rsq_curv_tol, newton_tol, newton_max_iters, n_threads,
            rsq, screen_beta, screen_is_active
        ),
        weights(weights.data(), weights.size()),
        y_mean(y_mean),
        y_var(y_var),
        screen_X_means(screen_X_means.data(), screen_X_means.size()),
        rsq_tol(rsq_tol),
        X(&X),
        resid(resid.data(), resid.size()),
        resid_sum(resid_sum),
        screen_grad(screen_beta.size())
    {
        resids.reserve(lmda_path.size());
        resid_sums.reserve(lmda_path.size());
    }
};

} // namespace state
} // namespace adelie_core