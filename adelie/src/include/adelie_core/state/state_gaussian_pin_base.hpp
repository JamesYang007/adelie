#pragma once
#include <vector>
#include <numeric>
#include <adelie_core/util/types.hpp>

#ifndef ADELIE_CORE_STATE_GAUSSIAN_PIN_BASE_TP
#define ADELIE_CORE_STATE_GAUSSIAN_PIN_BASE_TP \
    template <\
        class ConstraintType,\
        class ValueType,\
        class IndexType,\
        class BoolType\
    >
#endif
#ifndef ADELIE_CORE_STATE_GAUSSIAN_PIN_BASE
#define ADELIE_CORE_STATE_GAUSSIAN_PIN_BASE \
    StateGaussianPinBase<\
        ConstraintType,\
        ValueType,\
        IndexType,\
        BoolType\
    >
#endif

namespace adelie_core {
namespace state {

template <
    class ConstraintType,
    class ValueType=typename std::decay_t<ConstraintType>::value_t,
    class IndexType=Eigen::Index,
    class BoolType=bool
>
class StateGaussianPinBase
{
public:
    using constraint_t = ConstraintType;
    using value_t = ValueType;
    using index_t = IndexType;
    using bool_t = BoolType;
    using vec_value_t = util::rowvec_type<value_t>;
    using vec_index_t = util::rowvec_type<index_t>;
    using vec_bool_t = util::rowvec_type<bool_t>;
    using sp_vec_value_t = util::sp_vec_type<value_t, Eigen::RowMajor, index_t>;
    using map_vec_value_t = Eigen::Map<vec_value_t>;
    using map_vec_index_t = Eigen::Map<vec_index_t>;
    using map_vec_bool_t = Eigen::Map<vec_bool_t>;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;
    using map_cvec_index_t = Eigen::Map<const vec_index_t>;
    using dyn_vec_constraint_t = std::vector<constraint_t*>;
    using dyn_vec_value_t = std::vector<value_t>;
    using dyn_vec_index_t = std::vector<index_t>;
    using dyn_vec_sp_vec_t = std::vector<sp_vec_value_t>;
    using dyn_vec_vec_value_t = std::vector<vec_value_t>;
    using dyn_vec_vec_bool_t = std::vector<vec_bool_t>;
    using dyn_vec_mat_value_t = std::vector<util::rowmat_type<value_t>>;

    /* static states */
    const dyn_vec_constraint_t* constraints;
    const map_cvec_index_t groups;
    const map_cvec_index_t group_sizes;
    const value_t alpha;
    const map_cvec_value_t penalty;
    const map_cvec_index_t screen_set;
    const map_cvec_index_t screen_begins;
    const map_cvec_value_t screen_vars;
    const dyn_vec_mat_value_t* screen_transforms;
    const map_cvec_value_t lmda_path;

    /* configurations */
    const size_t constraint_buffer_size;
    const bool intercept;
    const size_t max_active_size;
    const size_t max_iters;
    const value_t tol;
    const value_t adev_tol;
    const value_t ddev_tol;
    const value_t newton_tol;
    const size_t newton_max_iters;
    const size_t n_threads;

    /* dynamic states */
    value_t rsq;
    map_vec_value_t screen_beta;
    map_vec_bool_t screen_is_active;
    size_t active_set_size;
    map_vec_index_t active_set;
    dyn_vec_index_t active_begins;
    dyn_vec_index_t active_order;
    dyn_vec_sp_vec_t betas;
    dyn_vec_value_t intercepts;
    dyn_vec_value_t rsqs;
    dyn_vec_value_t lmdas;
    size_t iters = 0;

    /* diagnostics */
    std::vector<double> benchmark_screen;
    std::vector<double> benchmark_active;

private:
    void initialize();

public:
    virtual ~StateGaussianPinBase() =default;
    
    explicit StateGaussianPinBase(
        const dyn_vec_constraint_t& constraints,
        const Eigen::Ref<const vec_index_t>& groups, 
        const Eigen::Ref<const vec_index_t>& group_sizes,
        value_t alpha, 
        const Eigen::Ref<const vec_value_t>& penalty,
        const Eigen::Ref<const vec_index_t>& screen_set, 
        const Eigen::Ref<const vec_index_t>& screen_begins, 
        const Eigen::Ref<const vec_value_t>& screen_vars,
        const dyn_vec_mat_value_t& screen_transforms,
        const Eigen::Ref<const vec_value_t>& lmda_path, 
        size_t constraint_buffer_size,
        bool intercept,
        size_t max_active_size,
        size_t max_iters,
        value_t tol,
        value_t adev_tol,
        value_t ddev_tol,
        value_t newton_tol,
        size_t newton_max_iters,
        size_t n_threads,
        value_t rsq,
        Eigen::Ref<vec_value_t> screen_beta, 
        Eigen::Ref<vec_bool_t> screen_is_active,
        size_t active_set_size,
        Eigen::Ref<vec_index_t> active_set
    ): 
        constraints(&constraints),
        groups(groups.data(), groups.size()),
        group_sizes(group_sizes.data(), group_sizes.size()),
        alpha(alpha),
        penalty(penalty.data(), penalty.size()),
        screen_set(screen_set.data(), screen_set.size()),
        screen_begins(screen_begins.data(), screen_begins.size()),
        screen_vars(screen_vars.data(), screen_vars.size()),
        screen_transforms(&screen_transforms),
        lmda_path(lmda_path.data(), lmda_path.size()),
        constraint_buffer_size(constraint_buffer_size),
        intercept(intercept),
        max_active_size(max_active_size),
        max_iters(max_iters),
        tol(tol),
        adev_tol(adev_tol),
        ddev_tol(ddev_tol),
        newton_tol(newton_tol),
        newton_max_iters(newton_max_iters),
        n_threads(n_threads),
        rsq(rsq),
        screen_beta(screen_beta.data(), screen_beta.size()),
        screen_is_active(screen_is_active.data(), screen_is_active.size()),
        active_set_size(active_set_size),
        active_set(active_set.data(), active_set.size())
    {
        initialize();
    }
};

} // namespace state
} // namespace adelie_core