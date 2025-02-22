#pragma once
#include <numeric>
#include <adelie_core/configs.hpp>
#include <adelie_core/solver/solver_gaussian_pin_base.hpp>
#include <adelie_core/util/counting_iterator.hpp>
#include <adelie_core/util/exceptions.hpp>
#include <adelie_core/util/functional.hpp>
#include <adelie_core/util/stopwatch.hpp>

namespace adelie_core {
namespace solver {
namespace gaussian {
namespace pin {
namespace cov {

template <class ValueType, class IndexType>
struct GaussianPinCovBufferPack: public GaussianPinBufferPack<ValueType, IndexType>
{
    using base_t = GaussianPinBufferPack<ValueType, IndexType>;
    using value_t = typename base_t::value_t;
    using index_t = typename base_t::index_t;

    util::rowvec_type<index_t> buffer_index;
    util::rowvec_type<value_t> buffer_sg;
    std::vector<value_t> active_beta_diff;
    std::vector<index_t> active_beta_diff_indices;
    std::vector<value_t> active_beta_diff_ordered;

    GaussianPinCovBufferPack(
        size_t buffer1_size,
        size_t buffer2_size,
        size_t buffer3_size,
        size_t buffer4_size,
        size_t constraint_buffer_size,
        size_t buffer_index_size,
        size_t buffer_sg_size,
        size_t active_beta_size
    ): 
        base_t(
            buffer1_size,
            buffer2_size,
            buffer3_size,
            buffer4_size,
            constraint_buffer_size,
            active_beta_size
        ),
        buffer_index(buffer_index_size),
        buffer_sg(buffer_sg_size)
    {
        // allocate buffers for optimization
        active_beta_diff.reserve(active_beta_size);
        active_beta_diff_indices.reserve(active_beta_size);
        active_beta_diff_ordered.reserve(active_beta_size);
    }
};

template <class StateType>
inline void update_active_inactive_subset(
    StateType& state
)
{
    using state_t = std::decay_t<StateType>;
    using vec_bool_t = typename state_t::vec_bool_t;

    const auto& group_sizes = state.group_sizes;
    const auto& screen_set = state.screen_set;
    const auto& screen_subset_order = state.screen_subset_order;
    const auto& screen_subset_ordered = state.screen_subset_ordered;
    const auto& screen_is_active = state.screen_is_active;
    auto& screen_is_active_subset = state.screen_is_active_subset;
    auto& active_subset_order = state.active_subset_order;
    auto& active_subset_ordered = state.active_subset_ordered;
    auto& inactive_subset_order = state.inactive_subset_order;
    auto& inactive_subset_ordered = state.inactive_subset_ordered;

    // update screen_is_active_subset
    int n_processed = 0;
    for (int ss_idx = 0; ss_idx < screen_set.size(); ++ss_idx) {
        const auto ss = screen_set[ss_idx];
        const auto gs = group_sizes[ss];
        Eigen::Map<vec_bool_t>(
            screen_is_active_subset.data() + n_processed, gs
        ) = screen_is_active[ss_idx];
        n_processed += gs;
    }

    // update active/inactive subset order/ordered
    active_subset_order.clear();
    active_subset_ordered.clear();
    inactive_subset_order.clear();
    inactive_subset_ordered.clear();
    for (int i = 0; i < screen_subset_order.size(); ++i) {
        const auto ssoi = screen_subset_order[i];
        const auto sso = screen_subset_ordered[i];
        if (screen_is_active_subset[ssoi]) {
            active_subset_order.push_back(i);
            active_subset_ordered.push_back(sso);
        } else {
            inactive_subset_order.push_back(i);
            inactive_subset_ordered.push_back(sso);
        }
    }
}


template <
    class StateType, 
    class BufferPackType,
    class IndicesType, 
    class ValuesType
>
ADELIE_CORE_STRONG_INLINE
void update_screen_grad_screen(
    StateType& state,
    BufferPackType& buffer_pack,
    const IndicesType& indices,
    const ValuesType& values
)
{
    const auto& screen_subset_order = state.screen_subset_order;
    const auto& screen_subset_ordered = state.screen_subset_ordered;
    auto& A = *state.A;
    auto& screen_grad = state.screen_grad;
    
    auto buffer_sg = buffer_pack.buffer_sg.head(screen_subset_ordered.size());

    A.bmul(screen_subset_ordered, indices, values, buffer_sg);

    for (int i = 0; i < screen_subset_order.size(); ++i) {
        screen_grad[screen_subset_order[i]] -= buffer_sg[i];
    }
}

template <
    class StateType, 
    class BufferPackType,
    class IndicesType, 
    class ValuesType
>
ADELIE_CORE_STRONG_INLINE
void update_screen_grad_active(
    StateType& state,
    BufferPackType& buffer_pack,
    const IndicesType& indices,
    const ValuesType& values
)
{
    using state_t = typename std::decay_t<StateType>;
    using vec_index_t = typename state_t::vec_index_t;

    const auto& screen_subset_order = state.screen_subset_order;
    const auto& active_subset_order = state.active_subset_order;
    const auto& active_subset_ordered = state.active_subset_ordered;
    auto& A = *state.A;
    auto& screen_grad = state.screen_grad;

    auto buffer_sg = buffer_pack.buffer_sg.head(active_subset_ordered.size());
    const Eigen::Map<const vec_index_t> active_subset_ordered_view(
        active_subset_ordered.data(),
        active_subset_ordered.size()
    );
    A.bmul(active_subset_ordered_view, indices, values, buffer_sg);

    for (size_t i = 0; i < active_subset_order.size(); ++i) {
        screen_grad[screen_subset_order[active_subset_order[i]]] -= buffer_sg[i];
    }
}

template <
    class StateType, 
    class BufferPackType,
    class IndicesType, 
    class ValuesType
>
ADELIE_CORE_STRONG_INLINE
void update_screen_grad_inactive(
    StateType& state,
    BufferPackType& buffer_pack,
    const IndicesType& indices,
    const ValuesType& values
)
{
    using state_t = typename std::decay_t<StateType>;
    using vec_index_t = typename state_t::vec_index_t;

    const auto& screen_subset_order = state.screen_subset_order;
    const auto& inactive_subset_order = state.inactive_subset_order;
    const auto& inactive_subset_ordered = state.inactive_subset_ordered;
    auto& A = *state.A;
    auto& screen_grad = state.screen_grad;

    auto buffer_sg = buffer_pack.buffer_sg.head(inactive_subset_ordered.size());
    const Eigen::Map<const vec_index_t> inactive_subset_ordered_view(
        inactive_subset_ordered.data(),
        inactive_subset_ordered.size()
    );

    A.bmul(inactive_subset_ordered_view, indices, values, buffer_sg);

    for (size_t i = 0; i < inactive_subset_order.size(); ++i) {
        screen_grad[screen_subset_order[inactive_subset_order[i]]] -= buffer_sg[i];
    }
}

template <class StateType, class ABDiffType, class IndicesType, class ValuesType>
inline void sparsify_active_beta_diff(
    const StateType& state,
    const ABDiffType& ab_diff,
    IndicesType& indices,
    ValuesType& values
)
{
    using state_t = std::decay_t<StateType>;
    using vec_value_t = typename state_t::vec_value_t;
    using vec_index_t = typename state_t::vec_index_t;

    const auto& groups = state.groups;
    const auto& group_sizes = state.group_sizes;
    const auto& screen_set = state.screen_set;
    const auto& active_set = state.active_set;
    const auto& active_order = state.active_order;
    const auto& active_begins = state.active_begins;

    auto idxs_begin = indices.data();
    auto vals_begin = values.data();
    for (size_t i = 0; i < active_order.size(); ++i) {
        const auto ss_idx = active_set[active_order[i]];
        const auto group = screen_set[ss_idx];
        const auto group_size = group_sizes[group];
        Eigen::Map<vec_index_t> idxs_seg(idxs_begin, group_size);
        Eigen::Map<vec_value_t> vals_seg(vals_begin, group_size);
        idxs_seg = vec_index_t::LinSpaced(
            group_size, groups[group], groups[group] + group_size - 1
        );
        vals_seg = ab_diff.segment(active_begins[active_order[i]], group_size);
        idxs_begin += group_size;
        vals_begin += group_size;
    }        
}

template <
    class StateType, 
    class Iter,
    class ValueType, 
    class BufferPackType,
    class UpdateCoefficientG0Type,
    class UpdateCoefficientG1Type,
    class UpdateScreenGradType,
    class AdditionalStepType=util::no_op
>
ADELIE_CORE_STRONG_INLINE
void coordinate_descent(
    StateType&& state,
    Iter begin,
    Iter end,
    size_t lmda_idx,
    ValueType& convg_measure,
    BufferPackType& buffer_pack,
    UpdateCoefficientG0Type update_coordinate_g0_f,
    UpdateCoefficientG1Type update_coordinate_g1_f,
    UpdateScreenGradType update_screen_grad_f,
    AdditionalStepType additional_step=AdditionalStepType()
)
{
    using state_t = std::decay_t<StateType>;
    using index_t = typename state_t::index_t;
    using value_t = typename state_t::value_t;
    using vec_value_t = typename state_t::vec_value_t;

    const auto& penalty = state.penalty;
    const auto& screen_set = state.screen_set;
    const auto& screen_begins = state.screen_begins;
    const auto& screen_vars = state.screen_vars;
    const auto& screen_transforms = *state.screen_transforms;
    const auto& groups = state.groups;
    const auto& group_sizes = state.group_sizes;
    const auto alpha = state.alpha;
    const auto lmda = state.lmda_path[lmda_idx];
    auto& screen_beta = state.screen_beta;
    auto& screen_grad = state.screen_grad;
    auto& rsq = state.rsq;

    auto& buffer1 = buffer_pack.buffer1;
    auto& buffer2 = buffer_pack.buffer2;
    auto& buffer3 = buffer_pack.buffer3;
    auto& buffer4 = buffer_pack.buffer4;
    auto& constraint_buffer = buffer_pack.constraint_buffer;
    auto& buffer_index = buffer_pack.buffer_index;

    const auto l1 = lmda * alpha;
    const auto l2 = lmda * (1-alpha);

    convg_measure = 0;
    for (auto it = begin; it != end; ++it) {
        const auto ss_idx = *it;              // index to screen set
        const auto k = screen_set[ss_idx];    // actual group index
        const auto ss_value_begin = screen_begins[ss_idx]; // value begin index at ss_idx
        const auto gsize = group_sizes[k]; // group size  

        if (gsize == 1) {
            auto& ak = screen_beta[ss_value_begin]; // corresponding beta
            auto gk = screen_grad[ss_value_begin]; // corresponding gradient
            const auto A_kk = screen_vars[ss_value_begin];  // corresponding A diagonal 
            const auto pk = penalty[k];

            const auto ak_old = ak;

            gk += ak_old * A_kk;
            update_coordinate_g0_f(
                ss_idx, ak, A_kk, gk, l1 * pk, l2 * pk, 1, constraint_buffer
            );
            gk -= ak_old * A_kk;

            if (ak_old == ak) continue;

            auto del = ak - ak_old;

            update_convergence_measure(convg_measure, del, A_kk);

            update_rsq(rsq, del, A_kk, gk);

            // update gradient 
            util::rowvec_type<index_t, 1> indices; 
            util::rowvec_type<value_t, 1> values;
            indices[0] = groups[k];
            values[0] = del;
            update_screen_grad_f(state, buffer_pack, indices, values);

        } else {
            auto ak = screen_beta.segment(ss_value_begin, gsize); // corresponding beta
            auto gk = screen_grad.segment(ss_value_begin, gsize); // corresponding gradient
            const auto& Vk = screen_transforms[ss_idx]; // corresponding V in SVD of X_c
            const auto A_kk = screen_vars.segment(ss_value_begin, gsize);  // corresponding A diagonal 
            const auto pk = penalty[k];

            auto gk_transformed = buffer3.head(ak.size());
            gk_transformed.matrix() = (
                gk.matrix() * Vk
            );

            // save old beta in buffer with transformation
            auto ak_old = buffer4.head(ak.size());
            ak_old = ak;
            auto ak_old_transformed = buffer4.segment(ak.size(), ak.size());
            ak_old_transformed.matrix() = ak_old.matrix() * Vk; 
            auto ak_transformed = buffer4.segment(2 * ak.size(), ak.size());
            Eigen::Map<vec_value_t>(
                ak_transformed.data(),
                ak_transformed.size()
             ) = ak_old_transformed;

            // update group coefficients
            gk_transformed += A_kk * ak_old_transformed; 
            update_coordinate_g1_f(
                ss_idx, ak_transformed, A_kk, gk_transformed, l1 * pk, l2 * pk, Vk, constraint_buffer
            );
            gk_transformed -= A_kk * ak_old_transformed; 

            if ((ak_old_transformed - ak_transformed).matrix().norm() <=  
                Configs::dbeta_tol * std::sqrt(gsize)) continue;
            
            auto del_transformed = buffer1.head(ak.size());
            del_transformed = ak_transformed - ak_old_transformed;

            update_convergence_measure(convg_measure, del_transformed, A_kk);

            update_rsq(rsq, del_transformed, A_kk, gk_transformed);

            // update new coefficient
            ak.matrix() = ak_transformed.matrix() * Vk.transpose();

            // update gradient
            auto del = buffer2.head(ak.size());
            del = ak - ak_old;
            auto indices = buffer_index.head(ak.size());
            indices = util::rowvec_type<index_t>::LinSpaced(
                ak.size(), 
                groups[k], 
                groups[k] + ak.size() - 1
            );
            update_screen_grad_f(state, buffer_pack, indices, del);
        }

        additional_step(ss_idx);
    }
}

/**
 * Applies multiple blockwise coordinate descent on the active set.
 */
template <
    class StateType, 
    class BufferPackType, 
    class UpdateCoefficientG0Type,
    class UpdateCoefficientG1Type,
    class CUIType
>
inline void solve_active(
    StateType&& state,
    size_t lmda_idx,
    BufferPackType& buffer_pack,
    UpdateCoefficientG0Type update_coordinate_g0_f,
    UpdateCoefficientG1Type update_coordinate_g1_f,
    CUIType check_user_interrupt
)
{
    using state_t = std::decay_t<StateType>;
    using value_t = typename state_t::value_t;
    using vec_value_t = typename state_t::vec_value_t;
    using vec_index_t = typename state_t::vec_index_t;

    const auto& group_sizes = state.group_sizes;
    const auto& screen_set = state.screen_set;
    const auto& screen_begins = state.screen_begins;
    const auto& active_set_size = state.active_set_size;
    const auto& active_set = state.active_set;
    const auto& active_begins = state.active_begins;
    const auto& screen_beta = state.screen_beta;
    const auto tol = state.tol;
    const auto max_iters = state.max_iters;
    auto& iters = state.iters;

    auto& active_beta_diff = buffer_pack.active_beta_diff;
    auto& active_beta_diff_indices = buffer_pack.active_beta_diff_indices;
    auto& active_beta_diff_ordered = buffer_pack.active_beta_diff_ordered;

    // update sizes
    size_t active_beta_size = 0;
    if (active_set_size) {
        const auto last_idx = active_set_size-1;
        const auto last_group = screen_set[active_set[last_idx]];
        const auto group_size = group_sizes[last_group];
        active_beta_size = active_begins[last_idx] + group_size;
    }
    active_beta_diff.resize(active_beta_size);
    active_beta_diff_indices.resize(active_beta_size);
    active_beta_diff_ordered.resize(active_beta_size);

    Eigen::Map<vec_value_t> ab_diff_view(
        active_beta_diff.data(), 
        active_beta_diff.size()
    );
    
    // save old active beta
    for (size_t i = 0; i < active_set_size; ++i) {
        const auto ss_idx_group = active_set[i];
        const auto ss_group = screen_set[ss_idx_group];
        const auto ss_group_size = group_sizes[ss_group];
        const auto sb_begin = screen_begins[ss_idx_group];
        const auto sb = screen_beta.segment(sb_begin, ss_group_size);
        const auto ab_begin = active_begins[i];
        auto ab_diff_view_curr = ab_diff_view.segment(ab_begin, ss_group_size);
        ab_diff_view_curr = sb;
    }
    
    while (1) {
        check_user_interrupt();
        ++iters;
        value_t convg_measure;
        coordinate_descent(
            state, 
            active_set.data(), active_set.data() + active_set_size,
            lmda_idx, 
            convg_measure, 
            buffer_pack,
            update_coordinate_g0_f,
            update_coordinate_g1_f,
            [](auto& state, auto& buffer_pack, const auto& indices, const auto& values) { 
                update_screen_grad_active(state, buffer_pack, indices, values); 
            }
        );
        if (convg_measure < tol) break;
        if (iters >= max_iters) throw util::max_cds_error(lmda_idx);
    }
    
    // compute new active beta - old active beta
    for (size_t i = 0; i < active_set_size; ++i) {
        const auto ss_idx_group = active_set[i];
        const auto ss_group = screen_set[ss_idx_group];
        const auto ss_group_size = group_sizes[ss_group];
        const auto sb_begin = screen_begins[ss_idx_group];
        const auto sb = screen_beta.segment(sb_begin, ss_group_size);
        const auto ab_begin = active_begins[i];
        auto ab_diff_view_curr = ab_diff_view.segment(ab_begin, ss_group_size);
        ab_diff_view_curr = sb - ab_diff_view_curr;
    }

    /* update screen gradient for non-active screen variables */

    // optimization: if active set is empty or active set is the same as screen set.
    if ((ab_diff_view.size() == 0) ||
        (static_cast<size_t>(active_set_size) 
            == static_cast<size_t>(screen_set.size()))) return;

    // sparsify ab_diff_view_curr
    Eigen::Map<vec_index_t> ab_diff_indices_view(
        active_beta_diff_indices.data(), 
        active_beta_diff_indices.size()
    );
    Eigen::Map<vec_value_t> ab_diff_ordered_view(
        active_beta_diff_ordered.data(), 
        active_beta_diff_ordered.size()
    );
    sparsify_active_beta_diff(
        state,
        ab_diff_view,
        ab_diff_indices_view,
        ab_diff_ordered_view
    );

    update_screen_grad_inactive(
        state, 
        buffer_pack, 
        ab_diff_indices_view, 
        ab_diff_ordered_view
    );
}

template <
    class StateType,
    class UpdateCoefficientG0Type,
    class UpdateCoefficientG1Type,
    class CUIType = util::no_op
>
inline void solve(
    StateType&& state,
    UpdateCoefficientG0Type update_coordinate_g0_f,
    UpdateCoefficientG1Type update_coordinate_g1_f,
    CUIType check_user_interrupt = CUIType()
)
{
    using state_t = std::decay_t<StateType>;
    using value_t = typename state_t::value_t;
    using index_t = typename state_t::index_t;
    using sp_vec_value_t = typename state_t::sp_vec_value_t;
    using sw_t = util::Stopwatch;

    auto& A = *state.A;
    const auto& groups = state.groups;
    const auto& group_sizes = state.group_sizes;
    const auto& screen_set = state.screen_set;
    const auto& screen_beta = state.screen_beta;
    const auto& lmda_path = state.lmda_path;
    const auto constraint_buffer_size = state.constraint_buffer_size;
    const auto max_active_size = state.max_active_size;
    const auto tol = state.tol;
    const auto rdev_tol = state.rdev_tol;
    const auto max_iters = state.max_iters;
    auto& active_set_size = state.active_set_size;
    auto& active_set = state.active_set;
    auto& active_begins = state.active_begins;
    auto& active_order = state.active_order;
    auto& screen_is_active = state.screen_is_active;
    auto& betas = state.betas;
    auto& intercepts = state.intercepts;
    auto& rsqs = state.rsqs;
    auto& lmdas = state.lmdas;
    auto& rsq = state.rsq;
    auto& iters = state.iters;
    auto& benchmark_screen = state.benchmark_screen;
    auto& benchmark_active = state.benchmark_active;

    sw_t stopwatch;
    const auto p = A.cols();

    // buffers for the routine
    const auto max_group_size = group_sizes.maxCoeff();
    GaussianPinCovBufferPack<value_t, index_t> buffer_pack(
        max_group_size, 
        max_group_size, 
        max_group_size, 
        3 * max_group_size,
        constraint_buffer_size,
        max_group_size,
        screen_beta.size(), 
        screen_beta.size()
    );    

    // buffer to store final result
    auto& active_beta_indices = buffer_pack.active_beta_indices; 
    auto& active_beta_ordered = buffer_pack.active_beta_ordered;

    // compute number of active coefficients
    size_t active_beta_size = 0;
    if (active_set_size) {
        const auto last_idx = active_set_size-1;
        const auto last_group = screen_set[active_set[last_idx]];
        const auto group_size = group_sizes[last_group];
        active_beta_size = active_begins[last_idx] + group_size;
    }
    
    const auto add_active_set = [&](auto ss_idx) {
        if (!screen_is_active[ss_idx]) {
            if (active_set_size >= max_active_size) {
                throw util::adelie_core_solver_error("Maximum number of active groups reached.");
            }

            screen_is_active[ss_idx] = true;

            active_set[active_set_size] = ss_idx;
            ++active_set_size;
        }
    };

    const auto lasso_active_and_update = [&](size_t l) {
        solve_active(
            state, 
            l, 
            buffer_pack,
            update_coordinate_g0_f,
            update_coordinate_g1_f,
            check_user_interrupt
        );
    };

    for (int l = 0; l < lmda_path.size(); ++l) {
        double screen_time = 0;
        double active_time = 0;

        while (1) {
            stopwatch.start();
            lasso_active_and_update(l);
            active_time += stopwatch.elapsed();

            check_user_interrupt();
            ++iters;
            value_t convg_measure;
            const auto old_active_size = active_set_size;
            stopwatch.start();
            coordinate_descent(
                state,
                util::counting_iterator<size_t>(0),
                util::counting_iterator<size_t>(screen_set.size()),
                l, 
                convg_measure,
                buffer_pack,
                update_coordinate_g0_f,
                update_coordinate_g1_f,
                [](auto& state, auto& buffer_pack, const auto& indices, const auto& values) { 
                    update_screen_grad_screen(state, buffer_pack, indices, values); 
                },
                add_active_set
            );
            screen_time += stopwatch.elapsed();
            const bool new_active_added = (old_active_size < active_set_size);

            if (new_active_added) {
                active_begins.resize(active_set_size);
                for (size_t i = old_active_size; i < active_begins.size(); ++i) {
                    active_begins[i] = active_beta_size;
                    const auto curr_group = screen_set[active_set[i]];
                    const auto curr_size = group_sizes[curr_group];
                    active_beta_size += curr_size;
                }

                // update active_order
                // NOTE: this is different from naive version.
                // solve_active() requires active_order so it must be updated.
                active_order.resize(active_set_size);
                std::iota(
                    std::next(active_order.begin(), old_active_size), 
                    active_order.end(), 
                    old_active_size
                );
                std::sort(
                    active_order.begin(), active_order.end(),
                    [&](auto i, auto j) { 
                        return groups[screen_set[active_set[i]]] < groups[screen_set[active_set[j]]];
                    }
                );

                update_active_inactive_subset(state);
            }

            if (convg_measure < tol) break;
            if (iters >= max_iters) throw util::max_cds_error(l);
        }

        // order the active betas
        active_beta_indices.resize(active_beta_size);
        active_beta_ordered.resize(active_beta_size);
        sparsify_active_beta(
            state,
            active_beta_indices,
            active_beta_ordered
        );

        Eigen::Map<const sp_vec_value_t> beta_map(
            p,
            active_beta_indices.size(),
            active_beta_indices.data(),
            active_beta_ordered.data()
        );

        sp_vec_value_t beta = beta_map;
        betas.emplace_back(std::move(beta));
        intercepts.emplace_back(0);
        rsqs.emplace_back(rsq);
        lmdas.emplace_back(lmda_path[l]);
        benchmark_screen.emplace_back(screen_time);
        benchmark_active.emplace_back(active_time);

        if ((l >= 1) && (rsqs[l]-rsqs[l-1] <= rdev_tol * rsqs[l])) break;
    }
}

template <class StateType, class CUIType = util::no_op>
inline void solve(
    StateType&& state,
    CUIType check_user_interrupt = CUIType()
)
{
    using state_t = std::decay_t<StateType>;
    using value_t = typename state_t::value_t;
    using vec_value_t = typename state_t::vec_value_t;

    const auto& constraints = *state.constraints;
    const auto& group_sizes = state.group_sizes;
    const auto& screen_set = state.screen_set;

    const auto max_group_size = group_sizes.maxCoeff();
    vec_value_t buff(max_group_size * 2);

    const auto update_coordinate_g0_f = [&](
        auto ss_idx, value_t& ak, value_t A_kk, value_t gk, value_t l1, value_t l2, value_t Q, auto& buffer
    ) {
        const auto k = screen_set[ss_idx];
        const auto constraint = constraints[k];

        // unconstrained case
        if (constraint == nullptr) {
            pin::update_coordinate(ak, A_kk, gk, l1, l2);

        // constrained case
        } else {
            Eigen::Map<util::rowvec_type<value_t, 1>> ak_view(&ak);
            const Eigen::Map<const util::rowvec_type<value_t, 1>> A_kk_view(&A_kk);
            const Eigen::Map<const util::rowvec_type<value_t, 1>> gk_view(&gk);
            const Eigen::Map<const util::colmat_type<value_t, 1, 1>> Q_view(&Q);
            constraint->solve(ak_view, A_kk_view, gk_view, l1, l2, Q_view, buffer);
        }
    };
    const auto update_coordinate_g1_f = [&](
        auto ss_idx, auto& ak, const auto& A_kk, const auto& gk, auto l1, auto l2, const auto& Q, auto& buffer
    ) {
        const auto k = screen_set[ss_idx];
        const auto constraint = constraints[k];

        // unconstrained case
        if (constraint == nullptr) {
            const auto size = ak.size();
            Eigen::Map<vec_value_t> buffer1(buff.data(), size);
            Eigen::Map<vec_value_t> buffer2(buff.data() + max_group_size, size);
            pin::update_coordinate(
                ak, A_kk, gk, l1, l2,
                state.newton_tol, state.newton_max_iters,
                buffer1, buffer2
            );

        // constrained case
        } else {
            constraint->solve(ak, A_kk, gk, l1, l2, Q, buffer);
        }
    };

    solve(
        state,
        update_coordinate_g0_f,
        update_coordinate_g1_f,
        check_user_interrupt
    );
}

} // namespace cov    
} // namespace pin
} // namespace gaussian
} // namespace solver
} // namespace adelie_core