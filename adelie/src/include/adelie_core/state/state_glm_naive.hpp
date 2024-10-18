#pragma once
#include <adelie_core/glm/glm_base.hpp>
#include <adelie_core/state/state_base.hpp>
#include <adelie_core/util/functional.hpp>
#include <adelie_core/util/tqdm.hpp>

#ifndef ADELIE_CORE_STATE_GLM_NAIVE_TP
#define ADELIE_CORE_STATE_GLM_NAIVE_TP \
    template <\
        class ConstraintType,\
        class MatrixType,\
        class ValueType,\
        class IndexType,\
        class BoolType,\
        class SafeBoolType\
    >
#endif
#ifndef ADELIE_CORE_STATE_GLM_NAIVE
#define ADELIE_CORE_STATE_GLM_NAIVE \
    StateGlmNaive<\
        ConstraintType,\
        MatrixType,\
        ValueType,\
        IndexType,\
        BoolType,\
        SafeBoolType\
    >
#endif

namespace adelie_core {
namespace state {

template <
    class ConstraintType,
    class MatrixType, 
    class ValueType=typename std::decay_t<MatrixType>::value_t,
    class IndexType=Eigen::Index,
    class BoolType=bool,
    class SafeBoolType=int8_t
>
class StateGlmNaive: public StateBase<
    ConstraintType,
    ValueType,
    IndexType,
    BoolType,
    SafeBoolType
>
{
public:
    using base_t = StateBase<
        ConstraintType,
        ValueType,
        IndexType,
        BoolType,
        SafeBoolType
    >;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::vec_bool_t;
    using typename base_t::map_cvec_value_t;
    using typename base_t::dyn_vec_constraint_t;
    using matrix_t = MatrixType;
    using glm_t = glm::GlmBase<value_t>;

    /* static states */
    const value_t loss_full;
    const map_cvec_value_t offsets;

    // convergence configs
    const size_t irls_max_iters;
    const value_t irls_tol;

    // other configs
    const bool setup_loss_null;

    /* dynamic states */
    value_t loss_null;
    matrix_t* X;

    // invariants
    value_t beta0;
    vec_value_t eta;
    vec_value_t resid;

private:
    void initialize();

public:
    explicit StateGlmNaive(
        matrix_t& X,
        const Eigen::Ref<const vec_value_t>& eta,
        const Eigen::Ref<const vec_value_t>& resid,
        const dyn_vec_constraint_t& constraints,
        const Eigen::Ref<const vec_index_t>& groups, 
        const Eigen::Ref<const vec_index_t>& group_sizes,
        const Eigen::Ref<const vec_index_t>& dual_groups, 
        value_t alpha, 
        const Eigen::Ref<const vec_value_t>& penalty,
        const Eigen::Ref<const vec_value_t>& offsets,
        const Eigen::Ref<const vec_value_t>& lmda_path,
        value_t loss_null,
        value_t loss_full,
        value_t lmda_max,
        value_t min_ratio,
        size_t lmda_path_size,
        size_t max_screen_size,
        size_t max_active_size,
        value_t pivot_subset_ratio,
        size_t pivot_subset_min,
        value_t pivot_slack_ratio,
        const std::string& screen_rule,
        size_t irls_max_iters,
        value_t irls_tol,
        size_t max_iters,
        value_t tol,
        value_t adev_tol,
        value_t ddev_tol,
        value_t newton_tol,
        size_t newton_max_iters,
        bool early_exit,
        bool setup_loss_null, 
        bool setup_lmda_max,
        bool setup_lmda_path,
        bool intercept,
        size_t n_threads,
        const Eigen::Ref<const vec_index_t>& screen_set,
        const Eigen::Ref<const vec_value_t>& screen_beta,
        const Eigen::Ref<const vec_bool_t>& screen_is_active,
        size_t active_set_size,
        const Eigen::Ref<const vec_index_t>& active_set,
        value_t beta0,
        value_t lmda,
        const Eigen::Ref<const vec_value_t>& grad
    ):
        base_t(
            constraints, groups, group_sizes, dual_groups, alpha, penalty, lmda_path, lmda_max, min_ratio, lmda_path_size,
            max_screen_size, max_active_size,
            pivot_subset_ratio, pivot_subset_min, pivot_slack_ratio, screen_rule, 
            max_iters, tol, adev_tol, ddev_tol, newton_tol, newton_max_iters, early_exit, 
            setup_lmda_max, setup_lmda_path, intercept, n_threads,
            screen_set, screen_beta, screen_is_active, active_set_size, active_set, lmda, grad
        ),
        loss_full(loss_full),
        offsets(offsets.data(), offsets.size()),
        irls_max_iters(irls_max_iters),
        irls_tol(irls_tol),
        setup_loss_null(setup_loss_null),
        loss_null(loss_null),
        X(&X),
        beta0(beta0),
        eta(eta),
        resid(resid)
    {
        initialize();
    }

    void solve(
        glm_t& glm,
        util::tq::progress_bar_t& pb,
        std::function<bool()> exit_cond,
        std::function<void()> check_user_interrupt =util::no_op()
    );
};

} // namespace state
} // namespace adelie_core