#pragma once
#include <vector>
#include <string>
#include <unordered_set>
#include <adelie_core/util/types.hpp>

namespace adelie_core {
namespace state {

/**
 * Updates absolute gradient in the base state.
 * The state DOES NOT have to be in its invariance. 
 * After the function finishes, abs_grad will reflect the correct value
 * respective to grad.
 */
template <class StateType>
ADELIE_CORE_STRONG_INLINE
void update_abs_grad(
    StateType& state
)
{
    const auto& grad = state.grad;
    const auto& groups = state.groups;
    const auto& group_sizes = state.group_sizes;
    const auto n_threads = state.n_threads;
    auto& abs_grad = state.abs_grad;

    const auto n_threads_capped = std::min<size_t>(n_threads, groups.size());
    #pragma omp parallel for schedule(static) num_threads(n_threads_capped)
    for (int i = 0; i < groups.size(); ++i) 
    {
        const auto k = groups[i];
        const auto size_k = group_sizes[i];
        abs_grad[i] = grad.segment(k, size_k).matrix().norm();
    }
}

/**
 * Updates all derived quantities from strong_set in the base class. 
 * The state must be such that only the strong_set is either unchanged from invariance,
 * or appended with new groups.
 * After the function finishes, all strong quantities in the base class
 * will be consistent with strong_set, and the state is otherwise effectively
 * unchanged in the sense that other quantities dependent on strong states are unchanged.
 */
template <class StateType>
ADELIE_CORE_STRONG_INLINE
void update_strong_derived_base(
    StateType& state
)
{
    const auto& groups = state.groups;
    const auto& group_sizes = state.group_sizes;
    const auto& strong_set = state.strong_set;
    auto& strong_hashset = state.strong_hashset;
    auto& strong_g1 = state.strong_g1;
    auto& strong_g2 = state.strong_g2;
    auto& strong_begins = state.strong_begins;
    auto& strong_order = state.strong_order;
    auto& strong_beta = state.strong_beta;
    auto& strong_is_active = state.strong_is_active;

    /* update strong_hashset */
    const auto old_strong_size = strong_begins.size();
    const auto strong_set_new_begin = std::next(strong_set.begin(), old_strong_size);
    strong_hashset.insert(strong_set_new_begin, strong_set.end());

    /* update strong_g1, strong_g2, strong_begins */
    size_t strong_value_size = (
        (old_strong_size == 0) ? 
        0 : (strong_begins.back() + group_sizes[strong_set[old_strong_size-1]])
    );
    for (size_t i = old_strong_size; i < strong_set.size(); ++i) {
        const auto curr_size = group_sizes[strong_set[i]];
        if (curr_size == 1) {
            strong_g1.push_back(i);
        } else {
            strong_g2.push_back(i);
        }
        strong_begins.push_back(strong_value_size);
        strong_value_size += curr_size;
    }

    /* update strong_order */
    strong_order.resize(strong_set.size());
    std::iota(
        std::next(strong_order.begin(), old_strong_size), 
        strong_order.end(), 
        old_strong_size
    );
    std::sort(
        strong_order.begin(),
        strong_order.end(),
        [&](auto i, auto j) {
            return groups[strong_set[i]] < groups[strong_set[j]];
        }
    );

    /* update strong_beta */
    strong_beta.resize(strong_value_size, 0);

    /* update strong_is_active */
    strong_is_active.resize(strong_set.size(), false);
}

enum class strong_rule_type
{
    _default,
    _fixed_greedy,
    _safe
};

template <class ValueType,
          class IndexType=Eigen::Index,
          class BoolType=bool
        >
struct StateBasilBase
{
    using value_t = ValueType;
    using index_t = IndexType;
    using bool_t = BoolType;
    using safe_bool_t = unsigned char;
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

    /* configurations */
    // lambda path configs
    const value_t min_ratio;
    const size_t lmda_path_size;

    // basil iteration configs
    const size_t delta_lmda_path_size;
    const size_t delta_strong_size;
    const size_t max_strong_size;
    const strong_rule_type strong_rule;

    // convergence configs
    const size_t max_iters;
    const value_t tol;
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
    uset_index_t strong_hashset;
    dyn_vec_index_t strong_set; 
    dyn_vec_index_t strong_g1;
    dyn_vec_index_t strong_g2;
    dyn_vec_index_t strong_begins;
    dyn_vec_index_t strong_order;
    dyn_vec_value_t strong_beta;
    dyn_vec_bool_t strong_is_active;
    value_t rsq;
    value_t lmda;
    vec_value_t grad;
    vec_value_t abs_grad;
    dyn_vec_sp_vec_t betas;
    dyn_vec_value_t intercepts;
    dyn_vec_value_t rsqs;
    dyn_vec_value_t lmdas;

    /* diagnostics */
    // TODO: fill

    virtual ~StateBasilBase() =default;

    explicit StateBasilBase(
        const Eigen::Ref<const vec_index_t>& groups, 
        const Eigen::Ref<const vec_index_t>& group_sizes,
        value_t alpha, 
        const Eigen::Ref<const vec_value_t>& penalty,
        const Eigen::Ref<const vec_value_t>& lmda_path,
        value_t lmda_max,
        value_t min_ratio,
        size_t lmda_path_size,
        size_t delta_lmda_path_size,
        size_t delta_strong_size,
        size_t max_strong_size,
        const std::string& strong_rule,
        size_t max_iters,
        value_t tol,
        value_t rsq_slope_tol,
        value_t rsq_curv_tol,
        value_t newton_tol,
        size_t newton_max_iters,
        bool early_exit,
        bool setup_lmda_max,
        bool setup_lmda_path,
        bool intercept,
        size_t n_threads,
        const Eigen::Ref<const vec_index_t>& strong_set,
        const Eigen::Ref<const vec_value_t>& strong_beta,
        const Eigen::Ref<const vec_bool_t>& strong_is_active,
        value_t rsq,
        value_t lmda,
        const Eigen::Ref<const vec_value_t>& grad
    ): 
        groups(groups.data(), groups.size()),
        group_sizes(group_sizes.data(), group_sizes.size()),
        alpha(alpha),
        penalty(penalty.data(), penalty.size()),
        min_ratio(min_ratio),
        lmda_path_size(lmda_path_size),
        delta_lmda_path_size(delta_lmda_path_size),
        delta_strong_size(delta_strong_size),
        max_strong_size(max_strong_size),
        strong_rule(convert_strong_rule(strong_rule)),
        max_iters(max_iters),
        tol(tol),
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
        strong_set(strong_set.data(), strong_set.data() + strong_set.size()),
        strong_beta(strong_beta.data(), strong_beta.data() + strong_beta.size()),
        strong_is_active(strong_is_active.data(), strong_is_active.data() + strong_is_active.size()),
        rsq(rsq),
        lmda(lmda),
        grad(grad),
        abs_grad(groups.size())
    {
        initialize();
    }

    strong_rule_type convert_strong_rule(
        const std::string& rule
    )
    {
        if (rule == "default") return strong_rule_type::_default;
        if (rule == "fixed_greedy") return strong_rule_type::_fixed_greedy;
        if (rule == "safe") return strong_rule_type::_safe;
        throw std::runtime_error("Invalid strong rule type: " + rule);
    }

    void initialize()
    {
        /* initialize strong_set derived quantities */
        strong_begins.reserve(strong_set.size());
        strong_g1.reserve(strong_set.size());
        strong_g2.reserve(strong_set.size());
        strong_begins.reserve(strong_set.size());
        strong_order.reserve(strong_set.size());
        update_strong_derived_base(*this);

        /* initialize abs_grad */
        update_abs_grad(*this);

        /* optimize for output storage size */
        const auto n_lmdas = std::max<size_t>(lmda_path.size(), lmda_path_size);
        betas.reserve(n_lmdas);
        intercepts.reserve(n_lmdas);
        rsqs.reserve(n_lmdas);
        lmdas.reserve(n_lmdas);
    }
};

} // namespace state
} // namespace adelie_core