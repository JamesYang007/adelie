#pragma once
#include <unordered_set>
#include <vector>
#include <adelie_core/solver/solver_base.hpp>
#include <adelie_core/util/exceptions.hpp>
#include <adelie_core/util/types.hpp>

#ifndef ADELIE_CORE_STATE_BASE_TP
#define ADELIE_CORE_STATE_BASE_TP \
    template <\
        class ConstraintType,\
        class ValueType,\
        class IndexType,\
        class BoolType,\
        class SafeBoolType\
    >
#endif
#ifndef ADELIE_CORE_STATE_BASE
#define ADELIE_CORE_STATE_BASE \
    StateBase<\
        ConstraintType,\
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
    class ValueType=typename std::decay_t<ConstraintType>::value_t,
    class IndexType=Eigen::Index,
    class BoolType=bool,
    class SafeBoolType=int8_t
>
class StateBase
{
public:
    using constraint_t = ConstraintType;
    using value_t = ValueType;
    using index_t = IndexType;
    using bool_t = BoolType;
    using safe_bool_t = SafeBoolType;
    using uset_index_t = std::unordered_set<index_t>;
    using vec_value_t = util::rowvec_type<value_t>;
    using vec_index_t = util::rowvec_type<index_t>;
    using vec_bool_t = util::rowvec_type<bool_t>;
    using vec_safe_bool_t = util::rowvec_type<safe_bool_t>;
    using sp_vec_value_t = util::sp_vec_type<value_t, Eigen::RowMajor, index_t>;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;
    using map_cvec_index_t = Eigen::Map<const vec_index_t>;
    using dyn_vec_constraint_t = std::vector<constraint_t*>;
    using dyn_vec_value_t = std::vector<value_t>;
    using dyn_vec_index_t = std::vector<index_t>;
    using dyn_vec_bool_t = std::vector<safe_bool_t>;
    using dyn_vec_sp_vec_t = std::vector<sp_vec_value_t>;

    /* static states */
    const dyn_vec_constraint_t constraints;
    const map_cvec_index_t groups;
    const map_cvec_index_t group_sizes;
    const map_cvec_index_t dual_groups;
    const value_t alpha;
    const map_cvec_value_t penalty;

    /* configurations */
    const size_t constraint_buffer_size;

    // lambda path configs
    const value_t min_ratio;
    const size_t lmda_path_size;

    // iteration configs
    const size_t max_screen_size;
    const size_t max_active_size;
    const value_t pivot_subset_ratio;
    const size_t pivot_subset_min;
    const value_t pivot_slack_ratio;
    const util::screen_rule_type screen_rule;

    // convergence configs
    const size_t max_iters;
    const value_t tol;
    const value_t adev_tol;
    const value_t ddev_tol;
    const value_t newton_tol;
    const size_t newton_max_iters;
    const bool early_exit;

    // other configs
    const bool setup_lmda_max;
    const bool setup_lmda_path;
    const bool intercept;
    const size_t n_threads;

    /* dynamic states */
    value_t lmda_max;
    vec_value_t lmda_path;

    // invariants
    uset_index_t screen_hashset;
    dyn_vec_index_t screen_set; 
    dyn_vec_index_t screen_begins;
    dyn_vec_value_t screen_beta;
    dyn_vec_bool_t screen_is_active;
    size_t active_set_size;
    vec_index_t active_set;
    value_t lmda;
    vec_value_t grad;
    vec_value_t abs_grad;

    // final results
    dyn_vec_sp_vec_t betas;
    dyn_vec_sp_vec_t duals;
    dyn_vec_value_t intercepts;
    dyn_vec_value_t devs;
    dyn_vec_value_t lmdas;

    // diagnostics
    std::vector<double> benchmark_screen;
    std::vector<double> benchmark_fit_screen;
    std::vector<double> benchmark_fit_active;
    std::vector<double> benchmark_kkt;
    std::vector<double> benchmark_invariance;
    std::vector<int> n_valid_solutions;
    std::vector<int> active_sizes;
    std::vector<int> screen_sizes;

private:
    inline void initialize();

public:
    virtual ~StateBase() =default;

    explicit StateBase(
        const dyn_vec_constraint_t& constraints,
        const Eigen::Ref<const vec_index_t>& groups, 
        const Eigen::Ref<const vec_index_t>& group_sizes,
        const Eigen::Ref<const vec_index_t>& dual_groups, 
        value_t alpha, 
        const Eigen::Ref<const vec_value_t>& penalty,
        const Eigen::Ref<const vec_value_t>& lmda_path,
        value_t lmda_max,
        value_t min_ratio,
        size_t lmda_path_size,
        size_t max_screen_size,
        size_t max_active_size,
        value_t pivot_subset_ratio,
        size_t pivot_subset_min,
        value_t pivot_slack_ratio,
        const std::string& screen_rule,
        size_t max_iters,
        value_t tol,
        value_t adev_tol,
        value_t ddev_tol,
        value_t newton_tol,
        size_t newton_max_iters,
        bool early_exit,
        bool setup_lmda_max,
        bool setup_lmda_path,
        bool intercept,
        size_t n_threads,
        const Eigen::Ref<const vec_index_t>& screen_set,
        const Eigen::Ref<const vec_value_t>& screen_beta,
        const Eigen::Ref<const vec_bool_t>& screen_is_active,
        size_t active_set_size,
        const Eigen::Ref<const vec_index_t>& active_set,
        value_t lmda,
        const Eigen::Ref<const vec_value_t>& grad
    ):
        constraints(constraints),
        groups(groups.data(), groups.size()),
        group_sizes(group_sizes.data(), group_sizes.size()),
        dual_groups(dual_groups.data(), dual_groups.size()),
        alpha(alpha),
        penalty(penalty.data(), penalty.size()),
        constraint_buffer_size(
            util::rowvec_type<size_t>::NullaryExpr(
                constraints.size(), 
                [&](auto i) { return constraints[i] ? constraints[i]->buffer_size() : 0; }
            ).maxCoeff()
        ),
        min_ratio(min_ratio),
        lmda_path_size(lmda_path_size),
        max_screen_size(max_screen_size),
        max_active_size(max_active_size),
        pivot_subset_ratio(pivot_subset_ratio),
        pivot_subset_min(pivot_subset_min),
        pivot_slack_ratio(pivot_slack_ratio),
        screen_rule(util::convert_screen_rule(screen_rule)),
        max_iters(max_iters),
        tol(tol),
        adev_tol(adev_tol),
        ddev_tol(ddev_tol),
        newton_tol(newton_tol),
        newton_max_iters(newton_max_iters),
        early_exit(early_exit),
        setup_lmda_max(setup_lmda_max),
        setup_lmda_path(setup_lmda_path),
        intercept(intercept),
        n_threads(n_threads),
        lmda_max(lmda_max),
        lmda_path(lmda_path),
        screen_set(screen_set.data(), screen_set.data() + screen_set.size()),
        screen_beta(screen_beta.data(), screen_beta.data() + screen_beta.size()),
        screen_is_active(screen_is_active.data(), screen_is_active.data() + screen_is_active.size()),
        active_set_size(active_set_size),
        active_set(active_set),
        lmda(lmda),
        grad(grad),
        abs_grad(groups.size())
    {
        initialize();
    }
};

ADELIE_CORE_STATE_BASE_TP
void
ADELIE_CORE_STATE_BASE::initialize()
{
    // sanity checks
    const auto G = groups.size();
    if (constraints.size() != static_cast<size_t>(G)) {
        throw util::adelie_core_error("constraints must be (G,) where groups is (G,).");
    }
    if (group_sizes.size() != G) {
        throw util::adelie_core_error("group_sizes must be (G,) where groups is (G,).");
    }
    if (dual_groups.size() != G) {
        throw util::adelie_core_error("dual_groups must be (G,) where groups is (G,).");
    }
    if (penalty.size() != G) {
        throw util::adelie_core_error("penalty must be (G,) where groups is (G,).");
    }
    if (alpha < 0 || alpha > 1) {
        throw util::adelie_core_error("alpha must be in [0,1].");
    }
    if (tol < 0) {
        throw util::adelie_core_error("tol must be >= 0.");
    }
    if (adev_tol < 0 || adev_tol > 1) {
        throw util::adelie_core_error("adev_tol must be in [0,1].");
    }
    if (ddev_tol < 0 || ddev_tol > 1) {
        throw util::adelie_core_error("ddev_tol must be in [0,1].");
    }
    if (newton_tol < 0) {
        throw util::adelie_core_error("newton_tol must be >= 0.");
    }
    if (n_threads < 1) {
        throw util::adelie_core_error("n_threads must be >= 1.");
    }
    if (min_ratio < 0 || min_ratio > 1) {
        throw util::adelie_core_error("min_ratio must be in [0,1].");
    }
    if (pivot_subset_ratio <= 0 || pivot_subset_ratio > 1) {
        throw util::adelie_core_error("pivot_subset_ratio must be in (0,1].");
    }
    if (pivot_subset_min < 1) {
        throw util::adelie_core_error("pivot_subset_min must be >= 1.");
    }
    if (pivot_slack_ratio < 0) {
        throw util::adelie_core_error("pivot_slack_ratio must be >= 0.");
    }
    if (screen_set.size() != screen_is_active.size()) {
        throw util::adelie_core_error("screen_is_active must be (s,) where screen_set is (s,).");
    }
    if (screen_beta.size() < screen_set.size()) {
        throw util::adelie_core_error(
            "screen_beta must be (bs,) where bs >= s and screen_set is (s,). "
            "It is likely screen_beta has been initialized incorrectly. "
        );
    }
    if (active_set_size > static_cast<size_t>(G)) {
        throw util::adelie_core_error(
            "active_set_size must be <= G where groups is (G,)."
        );
    }
    if (active_set.size() != G) {
        throw util::adelie_core_error(
            "active_set must be (G,) where groups is (G,)."
        );
    }
    if (grad.size() != groups[G-1] + group_sizes[G-1]) {
        throw util::adelie_core_error(
            "grad.size() != groups[G-1] + group_sizes[G-1]. "
            "It is likely either grad has the wrong shape, "
            "or groups/group_sizes have been initialized incorrectly."
        );
    }

    /* initialize screen_set derived quantities */
    screen_begins.reserve(screen_set.size());
    solver::update_screen_derived_base(*this);

    /* initialize abs_grad */
    solver::update_abs_grad(*this, lmda);

    /* optimize for output storage size */
    const auto n_lmdas = std::max<size_t>(lmda_path.size(), lmda_path_size);
    betas.reserve(n_lmdas);
    duals.reserve(n_lmdas);
    intercepts.reserve(n_lmdas);
    devs.reserve(n_lmdas);
    lmdas.reserve(n_lmdas);
    benchmark_fit_screen.reserve(n_lmdas);
    benchmark_fit_active.reserve(n_lmdas);
    benchmark_kkt.reserve(n_lmdas);
    benchmark_screen.reserve(n_lmdas);
    benchmark_invariance.reserve(n_lmdas);
    n_valid_solutions.reserve(n_lmdas);
    active_sizes.reserve(n_lmdas);
    screen_sizes.reserve(n_lmdas);
}

} // namespace state
} // namespace adelie_core