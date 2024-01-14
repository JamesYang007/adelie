#pragma once
#include <vector>
#include <numeric>
#include <adelie_core/util/types.hpp>

namespace adelie_core {
namespace state {

template <class ValueType,
          class IndexType=Eigen::Index,
          class BoolType=bool
        >
struct StateGaussianPinBase
{
    using value_t = ValueType;
    using index_t = IndexType;
    using bool_t = BoolType;
    using vec_value_t = util::rowvec_type<value_t>;
    using vec_index_t = util::rowvec_type<index_t>;
    using vec_bool_t = util::rowvec_type<bool_t>;
    using sp_vec_value_t = util::sp_vec_type<value_t, Eigen::RowMajor, index_t>;
    using map_vec_value_t = Eigen::Map<vec_value_t>;
    using map_vec_bool_t = Eigen::Map<vec_bool_t>;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;
    using map_cvec_index_t = Eigen::Map<const vec_index_t>;
    using dyn_vec_value_t = std::vector<value_t>;
    using dyn_vec_index_t = std::vector<index_t>;
    using dyn_vec_sp_vec_t = std::vector<sp_vec_value_t>;
    using dyn_vec_vec_value_t = std::vector<vec_value_t>;
    using dyn_vec_vec_bool_t = std::vector<vec_bool_t>;
    using dyn_vec_mat_value_t = std::vector<util::rowmat_type<value_t>>;

    /* static states */
    const map_cvec_index_t groups;
    const map_cvec_index_t group_sizes;
    const value_t alpha;
    const map_cvec_value_t penalty;
    const map_cvec_index_t screen_set;
    const map_cvec_index_t screen_g1;
    const map_cvec_index_t screen_g2;
    const map_cvec_index_t screen_begins;
    const map_cvec_value_t screen_vars;
    const dyn_vec_mat_value_t* screen_transforms;
    const map_cvec_value_t lmda_path;

    /* configurations */
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
    dyn_vec_index_t active_set;
    dyn_vec_index_t active_g1;
    dyn_vec_index_t active_g2;
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

    virtual ~StateGaussianPinBase() =default;
    
    explicit StateGaussianPinBase(
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
        Eigen::Ref<vec_bool_t> screen_is_active
    ): 
        groups(groups.data(), groups.size()),
        group_sizes(group_sizes.data(), group_sizes.size()),
        alpha(alpha),
        penalty(penalty.data(), penalty.size()),
        screen_set(screen_set.data(), screen_set.size()),
        screen_g1(screen_g1.data(), screen_g1.size()),
        screen_g2(screen_g2.data(), screen_g2.size()),
        screen_begins(screen_begins.data(), screen_begins.size()),
        screen_vars(screen_vars.data(), screen_vars.size()),
        screen_transforms(&screen_transforms),
        lmda_path(lmda_path.data(), lmda_path.size()),
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
        screen_is_active(screen_is_active.data(), screen_is_active.size())
    {
        active_set.reserve(screen_set.size());
        active_g1.reserve(screen_set.size());
        active_g2.reserve(screen_set.size());
        active_begins.reserve(screen_beta.size());
        int active_begin = 0;
        for (int i = 0; i < screen_is_active.size(); ++i) {
            if (!screen_is_active[i]) continue;
            active_set.push_back(i);
            int curr_size = group_sizes[screen_set[i]];
            if (curr_size == 1) {
                active_g1.push_back(i);
            } else {
                active_g2.push_back(i);
            }
            active_begins.push_back(active_begin);
            active_begin += curr_size;
        }

        active_order.resize(active_set.size());
        std::iota(active_order.begin(), active_order.end(), 0);
        std::sort(
            active_order.begin(),
            active_order.end(),
            [&](auto i, auto j) { 
                return groups[screen_set[active_set[i]]] < groups[screen_set[active_set[j]]]; 
            }
        );

        betas.reserve(lmda_path.size());
        intercepts.reserve(lmda_path.size());
        rsqs.reserve(lmda_path.size());
        lmdas.reserve(lmda_path.size());
        benchmark_screen.reserve(1000);
        benchmark_active.reserve(1000);
    }
};

} // namespace state
} // namespace adelie_core