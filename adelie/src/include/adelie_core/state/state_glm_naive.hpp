#pragma once
#include <adelie_core/state/state_glm_base.hpp>

namespace adelie_core {
namespace state {

template <class MatrixType, 
          class ValueType=typename std::decay_t<MatrixType>::value_t,
          class IndexType=Eigen::Index,
          class BoolType=bool
        >
struct StateGlmNaive : StateGlmBase<
        ValueType,
        IndexType,
        BoolType
    >
{
    using base_t = StateGlmBase<
        ValueType,
        IndexType,
        BoolType
    >;
    using typename base_t::value_t;
    using typename base_t::index_t;
    using typename base_t::uset_index_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::vec_bool_t;
    using typename base_t::map_cvec_value_t;
    using typename base_t::dyn_vec_value_t;
    using typename base_t::dyn_vec_index_t;
    using typename base_t::dyn_vec_bool_t;
    using umap_index_t = std::unordered_map<index_t, index_t>;
    using matrix_t = MatrixType;
    using dyn_vec_mat_value_t = std::vector<util::rowmat_type<value_t>>;

    /* static states */
    const vec_value_t weights_sqrt;
    const map_cvec_value_t X_means;
    const value_t y_mean;
    const value_t y_var;

    /* configurations */

    /* dynamic states */
    matrix_t* X;
    vec_value_t resid;
    value_t resid_sum;
    dyn_vec_value_t screen_X_means;
    dyn_vec_mat_value_t screen_transforms;
    dyn_vec_value_t screen_vars;

    /* buffers */
    vec_value_t resid_prev_valid;
    dyn_vec_value_t screen_beta_prev_valid; 
    dyn_vec_bool_t screen_is_active_prev_valid;

};

} // namespace state
} // namespace adelie_core