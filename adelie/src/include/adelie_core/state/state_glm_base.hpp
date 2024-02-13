#pragma once
#include <adelie_core/state/state_gaussian_base.hpp>
#include <adelie_core/util/types.hpp>

namespace adelie_core {
namespace state {

template <class ValueType,
          class IndexType=Eigen::Index,
          class BoolType=bool
        >
struct StateGlmBase
{
    using value_t = ValueType;
    using index_t = IndexType;
    using bool_t = BoolType;
    using safe_bool_t = int8_t;
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
    const value_t loss_full;
    const map_cvec_index_t groups;
    const map_cvec_index_t group_sizes;
    const value_t alpha;
    const map_cvec_value_t penalty;
    const map_cvec_value_t weights;
    const map_cvec_value_t offsets;

    /* configurations */
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
    const size_t irls_max_iters;
    const value_t irls_tol;
    const size_t max_iters;
    const value_t tol;
    const value_t adev_tol;
    const value_t ddev_tol;
    const value_t newton_tol;
    const size_t newton_max_iters;
    const bool early_exit;

    // other configs
    const bool setup_loss_null;
    const bool setup_lmda_max;
    const bool setup_lmda_path;
    const bool intercept;
    const size_t n_threads;

    /* dynamic states */
    value_t loss_null;
    value_t lmda_max;
    vec_value_t lmda_path;

    // invariants
    uset_index_t screen_hashset;
    dyn_vec_index_t screen_set; 
    dyn_vec_index_t screen_g1;
    dyn_vec_index_t screen_g2;
    dyn_vec_index_t screen_begins;
    dyn_vec_value_t screen_beta;
    dyn_vec_bool_t screen_is_active;
    value_t beta0;
    value_t lmda;
    vec_value_t grad;
    vec_value_t abs_grad;

    // final results
    dyn_vec_sp_vec_t betas;
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

    virtual ~StateGlmBase() =default;

    explicit StateGlmBase(
        const Eigen::Ref<const vec_index_t>& groups, 
        const Eigen::Ref<const vec_index_t>& group_sizes,
        value_t alpha, 
        const Eigen::Ref<const vec_value_t>& penalty,
        const Eigen::Ref<const vec_value_t>& weights,
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
        value_t beta0,
        value_t lmda,
        const Eigen::Ref<const vec_value_t>& grad
    ): 
        loss_full(loss_full),
        groups(groups.data(), groups.size()),
        group_sizes(group_sizes.data(), group_sizes.size()),
        alpha(alpha),
        penalty(penalty.data(), penalty.size()),
        weights(weights.data(), weights.size()),
        offsets(offsets.data(), offsets.size()),
        min_ratio(min_ratio),
        lmda_path_size(lmda_path_size),
        max_screen_size(max_screen_size),
        max_active_size(max_active_size),
        pivot_subset_ratio(pivot_subset_ratio),
        pivot_subset_min(pivot_subset_min),
        pivot_slack_ratio(pivot_slack_ratio),
        screen_rule(util::convert_screen_rule(screen_rule)),
        irls_max_iters(irls_max_iters),
        irls_tol(irls_tol),
        max_iters(max_iters),
        tol(tol),
        adev_tol(adev_tol),
        ddev_tol(ddev_tol),
        newton_tol(newton_tol),
        newton_max_iters(newton_max_iters),
        early_exit(early_exit),
        setup_loss_null(setup_loss_null),
        setup_lmda_max(setup_lmda_max),
        setup_lmda_path(setup_lmda_path),
        intercept(intercept),
        n_threads(n_threads),
        loss_null(loss_null),
        lmda_max(lmda_max),
        lmda_path(lmda_path),
        screen_set(screen_set.data(), screen_set.data() + screen_set.size()),
        screen_beta(screen_beta.data(), screen_beta.data() + screen_beta.size()),
        screen_is_active(screen_is_active.data(), screen_is_active.data() + screen_is_active.size()),
        beta0(beta0),
        lmda(lmda),
        grad(grad),
        abs_grad(groups.size())
    {
        initialize();
    }

    void initialize()
    {
        /* initialize screen_set derived quantities */
        screen_begins.reserve(screen_set.size());
        screen_g1.reserve(screen_set.size());
        screen_g2.reserve(screen_set.size());
        gaussian::update_screen_derived_base(*this);

        /* initialize abs_grad */
        gaussian::update_abs_grad(*this, lmda);

        /* optimize for output storage size */
        const auto n_lmdas = std::max<size_t>(lmda_path.size(), lmda_path_size);
        betas.reserve(n_lmdas);
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
};

} // namespace state
} // namespace adelie_core