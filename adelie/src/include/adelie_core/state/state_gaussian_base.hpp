#pragma once
#include <vector>
#include <string>
#include <unordered_set>
#include <adelie_core/util/types.hpp>
#include <adelie_core/util/macros.hpp>

namespace adelie_core {
namespace state {

template <class GroupsType, class GroupSizesType,
          class PenaltyType, class GradType, 
          class ScreenSetType, class ScreenHashsetType,
          class ScreenBeginsType, class ScreenBetaType,
          class ValueType, class AbsGradType>
void update_abs_grad(
    const GroupsType& groups,
    const GroupSizesType& group_sizes,
    const PenaltyType& penalty,
    const GradType& grad,
    const ScreenSetType& screen_set,
    const ScreenHashsetType& screen_hashset,
    const ScreenBeginsType& screen_begins,
    const ScreenBetaType& screen_beta,
    ValueType lmda,
    ValueType alpha,
    AbsGradType& abs_grad,
    size_t n_threads
)
{
    const auto is_screen = [&](auto i) {
        return screen_hashset.find(i) != screen_hashset.end();
    };

    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (int ss_idx = 0; ss_idx < screen_set.size(); ++ss_idx) 
    {
        const auto i = screen_set[ss_idx];
        const auto b = screen_begins[ss_idx]; 
        const auto k = groups[i];
        const auto size_k = group_sizes[i];
        const auto pk = penalty[i];
        abs_grad[i] = (
            grad.segment(k, size_k) - 
            ((1-alpha) * pk) * (lmda * Eigen::Map<const util::rowvec_type<ValueType>>(
                screen_beta.data() + b,
                size_k
            ))
        ).matrix().norm();
    }

    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (int i = 0; i < groups.size(); ++i) 
    {
        if (is_screen(i)) continue;
        const auto k = groups[i];
        const auto size_k = group_sizes[i];
        abs_grad[i] = grad.segment(k, size_k).matrix().norm();
    }
}

/**
 * Updates absolute gradient in the base state.
 * The state DOES NOT have to be in its invariance. 
 * After the function finishes, abs_grad will reflect the correct value
 * respective to grad.
 */
template <class StateType, class ValueType>
ADELIE_CORE_STRONG_INLINE
void update_abs_grad(
    StateType& state,
    ValueType lmda
)
{
    update_abs_grad(
        state.groups,
        state.group_sizes,
        state.penalty,
        state.grad,
        state.screen_set,
        state.screen_hashset,
        state.screen_begins,
        state.screen_beta,
        lmda,
        state.alpha,
        state.abs_grad,
        state.n_threads
    );
}

/**
 * Updates all derived quantities from screen_set in the base class. 
 * The state must be such that only the screen_set is either unchanged from invariance,
 * or appended with new groups.
 * After the function finishes, all strong quantities in the base class
 * will be consistent with screen_set, and the state is otherwise effectively
 * unchanged in the sense that other quantities dependent on strong states are unchanged.
 */
template <class StateType>
ADELIE_CORE_STRONG_INLINE
void update_screen_derived_base(
    StateType& state
)
{
    const auto& groups = state.groups;
    const auto& group_sizes = state.group_sizes;
    const auto& screen_set = state.screen_set;
    auto& screen_hashset = state.screen_hashset;
    auto& screen_g1 = state.screen_g1;
    auto& screen_g2 = state.screen_g2;
    auto& screen_begins = state.screen_begins;
    auto& screen_order = state.screen_order;
    auto& screen_beta = state.screen_beta;
    auto& screen_is_active = state.screen_is_active;

    /* update screen_hashset */
    const auto old_screen_size = screen_begins.size();
    const auto screen_set_new_begin = std::next(screen_set.begin(), old_screen_size);
    screen_hashset.insert(screen_set_new_begin, screen_set.end());

    /* update screen_g1, screen_g2, screen_begins */
    size_t screen_value_size = (
        (old_screen_size == 0) ? 
        0 : (screen_begins.back() + group_sizes[screen_set[old_screen_size-1]])
    );
    for (size_t i = old_screen_size; i < screen_set.size(); ++i) {
        const auto curr_size = group_sizes[screen_set[i]];
        if (curr_size == 1) {
            screen_g1.push_back(i);
        } else {
            screen_g2.push_back(i);
        }
        screen_begins.push_back(screen_value_size);
        screen_value_size += curr_size;
    }

    /* update screen_order */
    screen_order.resize(screen_set.size());
    std::iota(
        std::next(screen_order.begin(), old_screen_size), 
        screen_order.end(), 
        old_screen_size
    );
    std::sort(
        screen_order.begin(),
        screen_order.end(),
        [&](auto i, auto j) {
            return groups[screen_set[i]] < groups[screen_set[j]];
        }
    );

    /* update screen_beta */
    screen_beta.resize(screen_value_size, 0);

    /* update screen_is_active */
    screen_is_active.resize(screen_set.size(), false);
}

enum class screen_rule_type
{
    _strong,
    _pivot
};

template <class ValueType,
          class IndexType=Eigen::Index,
          class BoolType=bool
        >
struct StateGaussianBase
{
    using value_t = ValueType;
    using index_t = IndexType;
    using bool_t = BoolType;
    using safe_bool_t = int;
    using uset_index_t = std::unordered_set<index_t>;
    using vec_value_t = util::rowvec_type<value_t>;
    using vec_index_t = util::rowvec_type<index_t>;
    using vec_bool_t = util::rowvec_type<bool_t>;
    using vec_safe_bool_t = util::rowvec_type<safe_bool_t>;
    using sp_vec_value_t = util::sp_vec_type<value_t, Eigen::RowMajor, index_t>;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;
    using map_cvec_index_t = Eigen::Map<const vec_index_t>;
    using dyn_vec_value_t = std::vector<value_t>;
    using dyn_vec_index_t = std::vector<index_t>;
    using dyn_vec_bool_t = std::vector<safe_bool_t>;
    using dyn_vec_sp_vec_t = std::vector<sp_vec_value_t>;

    /* static states */
    const map_cvec_index_t groups;
    const map_cvec_index_t group_sizes;
    const value_t alpha;
    const map_cvec_value_t penalty;
    const map_cvec_value_t weights;

    /* configurations */
    // lambda path configs
    const value_t min_ratio;
    const size_t lmda_path_size;

    // iteration configs
    const size_t max_screen_size;
    const value_t pivot_subset_ratio;
    const size_t pivot_subset_min;
    const value_t pivot_slack_ratio;
    const screen_rule_type screen_rule;

    // convergence configs
    const size_t max_iters;
    const value_t tol;
    const value_t rsq_tol;
    const value_t rsq_slope_tol;
    const value_t rsq_curv_tol;
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
    dyn_vec_index_t screen_g1;
    dyn_vec_index_t screen_g2;
    dyn_vec_index_t screen_begins;
    dyn_vec_index_t screen_order;
    dyn_vec_value_t screen_beta;
    dyn_vec_bool_t screen_is_active;
    value_t rsq;
    value_t lmda;
    vec_value_t grad;
    vec_value_t abs_grad;

    // final results
    dyn_vec_sp_vec_t betas;
    dyn_vec_value_t intercepts;
    dyn_vec_value_t rsqs;
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

    virtual ~StateGaussianBase() =default;

    explicit StateGaussianBase(
        const Eigen::Ref<const vec_index_t>& groups, 
        const Eigen::Ref<const vec_index_t>& group_sizes,
        value_t alpha, 
        const Eigen::Ref<const vec_value_t>& penalty,
        const Eigen::Ref<const vec_value_t>& weights,
        const Eigen::Ref<const vec_value_t>& lmda_path,
        value_t lmda_max,
        value_t min_ratio,
        size_t lmda_path_size,
        size_t max_screen_size,
        value_t pivot_subset_ratio,
        size_t pivot_subset_min,
        value_t pivot_slack_ratio,
        const std::string& screen_rule,
        size_t max_iters,
        value_t tol,
        value_t rsq_tol,
        value_t rsq_slope_tol,
        value_t rsq_curv_tol,
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
        value_t rsq,
        value_t lmda,
        const Eigen::Ref<const vec_value_t>& grad
    ): 
        groups(groups.data(), groups.size()),
        group_sizes(group_sizes.data(), group_sizes.size()),
        alpha(alpha),
        penalty(penalty.data(), penalty.size()),
        weights(weights.data(), weights.size()),
        min_ratio(min_ratio),
        lmda_path_size(lmda_path_size),
        max_screen_size(max_screen_size),
        pivot_subset_ratio(pivot_subset_ratio),
        pivot_subset_min(pivot_subset_min),
        pivot_slack_ratio(pivot_slack_ratio),
        screen_rule(convert_screen_rule(screen_rule)),
        max_iters(max_iters),
        tol(tol),
        rsq_tol(rsq_tol),
        rsq_slope_tol(rsq_slope_tol),
        rsq_curv_tol(rsq_curv_tol),
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
        rsq(rsq),
        lmda(lmda),
        grad(grad),
        abs_grad(groups.size())
    {
        initialize();
    }

    screen_rule_type convert_screen_rule(
        const std::string& rule
    )
    {
        if (rule == "strong") return screen_rule_type::_strong;
        if (rule == "pivot") return screen_rule_type::_pivot;
        throw std::runtime_error("Invalid strong rule type: " + rule);
    }

    void initialize()
    {
        /* initialize screen_set derived quantities */
        screen_begins.reserve(screen_set.size());
        screen_g1.reserve(screen_set.size());
        screen_g2.reserve(screen_set.size());
        screen_begins.reserve(screen_set.size());
        screen_order.reserve(screen_set.size());
        update_screen_derived_base(*this);

        /* initialize abs_grad */
        update_abs_grad(*this, lmda);

        /* optimize for output storage size */
        const auto n_lmdas = std::max<size_t>(lmda_path.size(), lmda_path_size);
        betas.reserve(n_lmdas);
        intercepts.reserve(n_lmdas);
        rsqs.reserve(n_lmdas);
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
};

} // namespace state
} // namespace adelie_core