#pragma once
#include <adelie_core/state/state_gaussian_pin_base.hpp>

namespace adelie_core {
namespace state {

template <class MatrixType, 
          class ValueType=typename std::decay_t<MatrixType>::value_t,
          class IndexType=Eigen::Index,
          class BoolType=bool
        >
struct StateGaussianPinCov: StateGaussianPinBase<
        ValueType,
        IndexType,
        BoolType
    >
{
    using base_t = StateGaussianPinBase<
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
    using typename base_t::dyn_vec_vec_value_t;
    using matrix_t = MatrixType;

    /* configurations */
    const value_t rdev_tol;

    /* dynamic states */
    matrix_t* A;    // covariance matrix-like
    map_vec_value_t screen_grad;

    explicit StateGaussianPinCov(
        matrix_t& A,
        const Eigen::Ref<const vec_index_t>& groups, 
        const Eigen::Ref<const vec_index_t>& group_sizes,
        value_t alpha, 
        const Eigen::Ref<const vec_value_t>& penalty,
        const Eigen::Ref<const vec_index_t>& screen_set, 
        const Eigen::Ref<const vec_index_t>& screen_g1,
        const Eigen::Ref<const vec_index_t>& screen_g2,
        const Eigen::Ref<const vec_index_t>& screen_begins, 
        const Eigen::Ref<const vec_value_t>& screen_vars,
        const dyn_vec_mat_value_t& screen_transforms,
        const Eigen::Ref<const vec_value_t>& lmda_path, 
        size_t max_active_size,
        size_t max_iters,
        value_t tol,
        value_t rdev_tol,
        value_t newton_tol,
        size_t newton_max_iters,
        size_t n_threads,
        value_t rsq,
        Eigen::Ref<vec_value_t> screen_beta, 
        Eigen::Ref<vec_value_t> screen_grad,
        Eigen::Ref<vec_bool_t> screen_is_active
    ): 
        base_t(
            groups, group_sizes, alpha, penalty, 
            screen_set, screen_g1, screen_g2, screen_begins, screen_vars, screen_transforms, lmda_path, 
            false, max_active_size, max_iters, tol, 0, 0, newton_tol, newton_max_iters, n_threads,
            rsq, screen_beta, screen_is_active
        ),
        rdev_tol(rdev_tol),
        A(&A),
        screen_grad(screen_grad.data(), screen_grad.size())
    {}
};

} // namespace state
} // namespace adelie_core