#pragma once
#include <adelie_core/util/counting_iterator.hpp>
#include <adelie_core/util/functional.hpp>
#include <adelie_core/util/macros.hpp>
#include <adelie_core/util/stopwatch.hpp>
#include <adelie_core/util/types.hpp>

namespace adelie_core {
namespace optimization {

template <class MatrixType,
          class ValueType=typename std::decay_t<MatrixType>::value_t,
          class IndexType=Eigen::Index,
          class SafeBoolType=int8_t
        >
struct StateNNQPLowRank
{
    using matrix_t = MatrixType;
    using value_t = ValueType;
    using index_t = IndexType;
    using safe_bool_t = SafeBoolType;
    using uset_index_t = std::unordered_set<index_t>;
    using vec_value_t = util::rowvec_type<value_t>;
    using colmat_value_t = util::colmat_type<value_t>;
    using rowarr_value_t = util::rowarr_type<value_t>;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;
    using map_ccolmat_value_t = Eigen::Map<const colmat_value_t>;
    using dyn_vec_value_t = std::vector<value_t>;
    using dyn_vec_index_t = std::vector<index_t>;
    using dyn_vec_bool_t = std::vector<safe_bool_t>;

    /* static states */
    const value_t quad_c;
    const value_t quad_d;
    const map_cvec_value_t quad_alpha;
    const map_ccolmat_value_t quad_Sigma;
    const map_cvec_value_t linear_v;

    /* configurations */
    const size_t max_iters;
    const value_t tol;
    const util::nnqp_screen_rule_type screen_rule;

    /* dynamic states */
    size_t iters = 0;
    matrix_t* A; 
    uset_index_t screen_hashset;
    dyn_vec_index_t screen_set; 
    dyn_vec_value_t screen_beta;
    dyn_vec_bool_t screen_is_active;
    dyn_vec_value_t screen_vars;
    dyn_vec_value_t screen_ASigma;
    dyn_vec_index_t active_set;
    vec_value_t resid_A;
    value_t resid_alpha;
    vec_value_t grad;

    dyn_vec_value_t screen_grad;

    /* benchmark */
    double benchmark_screen = 0;
    double benchmark_invariance = 0;
    double benchmark_fit_screen = 0;
    double benchmark_fit_active = 0;
    double benchmark_kkt = 0;

    StateNNQPLowRank(
        matrix_t& A,
        value_t quad_c,
        value_t quad_d,
        const Eigen::Ref<const vec_value_t>& quad_alpha,
        const Eigen::Ref<const colmat_value_t>& quad_Sigma,
        const Eigen::Ref<const vec_value_t>& linear_v,
        size_t max_iters,
        value_t tol,
        const std::string& screen_rule
    ):
        quad_c(quad_c),
        quad_d(quad_d),
        quad_alpha(quad_alpha.data(), quad_alpha.size()),
        quad_Sigma(quad_Sigma.data(), quad_Sigma.rows(), quad_Sigma.cols()),
        linear_v(linear_v.data(), linear_v.size()),
        max_iters(max_iters),
        tol(tol),
        screen_rule(util::convert_nnqp_screen_rule(screen_rule)),
        A(&A),
        resid_A(vec_value_t::Zero(A.cols())),
        resid_alpha(0),
        grad(A.rows())
    {
        // TODO: optimize for storage?
        // Seems dangerous to reserve O(n) space since 
        // we're going to potentially need many of these objects.
    }

    void warm_start(StateNNQPLowRank&& state)
    {
        if (state.A != this->A) {
            throw util::adelie_core_error(
                "Warm start requires the A matrix to be the same object. "
            );
        }

        screen_hashset = std::move(state.screen_hashset);
        screen_set = std::move(state.screen_set); 
        screen_beta = std::move(state.screen_beta);
        screen_is_active = std::move(state.screen_is_active);
        update_screen_derived(*this);
        active_set = std::move(state.active_set);
        resid_A = std::move(state.resid_A);

        resid_alpha = 0;
        for (int i = 0; i < active_set.size(); ++i) {
            const auto ss_idx = active_set[i];
            const auto k = screen_set[ss_idx];
            const auto xk = screen_beta[ss_idx];
            resid_alpha += quad_alpha[k] * xk;
        }
    }
};

namespace nnqp_low_rank {
namespace pin {

template <class StateType, class IterType, class ValueType,
          class AdditionalStepType=util::no_op>
ADELIE_CORE_STRONG_INLINE
void coordinate_descent(
    StateType&& state,
    IterType begin,
    IterType end,
    ValueType& convg_measure,
    AdditionalStepType additional_step=AdditionalStepType()
)
{
    using state_t = std::decay_t<StateType>;
    using value_t = typename state_t::value_t;
    using rowarr_value_t = typename state_t::rowarr_value_t;

    const auto quad_c = state.quad_c;
    const auto quad_d = state.quad_d;
    const auto& quad_alpha = state.quad_alpha;
    const auto& quad_Sigma = state.quad_Sigma;
    const auto& linear_v = state.linear_v;
    const auto& screen_set = state.screen_set;
    const auto& screen_vars = state.screen_vars;
    const auto& screen_ASigma = state.screen_ASigma;
    auto& A = *state.A;
    auto& screen_beta = state.screen_beta;
    auto& screen_grad = state.screen_grad;
    auto& resid_A = state.resid_A;
    auto& resid_alpha = state.resid_alpha;

    const auto d = quad_Sigma.cols();
    const Eigen::Map<const rowarr_value_t> screen_ASigma_map(
        screen_ASigma.data(),
        screen_ASigma.size() / d,
        d
    );

    convg_measure = 0;
    for (auto it = begin; it != end; ++it) {
        const auto ss_idx = *it;
        const auto k = screen_set[ss_idx];
        const auto quad_alpha_k = quad_alpha[k];
        const auto H_kk = screen_vars[ss_idx];
        const auto AkSigma = screen_ASigma_map.row(ss_idx);
        const auto v_k = linear_v[k];
        auto& ak = screen_beta[ss_idx];
        auto& gk = screen_grad[ss_idx];

        const auto ak_old = ak;

        // compute gradient
        gk = v_k - quad_c * (
            quad_d * quad_alpha_k * resid_alpha +
            (AkSigma * resid_A).sum()
        );

        // update coefficient
        ak = std::max<value_t>(ak_old + gk / H_kk, 0.0);

        if (ak_old == ak) continue;

        const auto del = ak - ak_old;

        convg_measure = std::max<value_t>(convg_measure, del * del * H_kk);

        resid_alpha += quad_alpha_k * del;
        A.rtmul(k, del, resid_A);

        additional_step(ss_idx);
    }
}

template <class StateType, 
          class CUIType>
ADELIE_CORE_STRONG_INLINE
void solve_active(
    StateType&& state,
    CUIType check_user_interrupt
)
{
    using state_t = std::decay_t<StateType>;
    using value_t = typename state_t::value_t;

    const auto& active_set = state.active_set;
    const auto tol = state.tol;
    const auto max_iters = state.max_iters;
    auto& iters = state.iters;

    while (1) {
        check_user_interrupt();
        ++iters;
        value_t convg_measure;
        coordinate_descent(
            state, 
            active_set.data(), active_set.data() + active_set.size(),
            convg_measure
        );
        if (convg_measure < tol) break;
        if (iters >= max_iters) {
            throw util::adelie_core_solver_error(
                "Maximum number of iterations reached."
            );
        }
    }
}

template <class StateType,
          class CUIType = util::no_op>
inline void solve(
    StateType&& state,
    CUIType check_user_interrupt = CUIType()
)
{
    using state_t = std::decay_t<StateType>;
    using value_t = typename state_t::value_t;
    using index_t = typename state_t::index_t;
    using sw_t = util::Stopwatch;

    const auto tol = state.tol;
    const auto max_iters = state.max_iters;
    const auto& screen_set = state.screen_set;
    auto& screen_is_active = state.screen_is_active;
    auto& active_set = state.active_set;
    auto& iters = state.iters;
    auto& screen_time = state.benchmark_fit_screen;
    auto& active_time = state.benchmark_fit_active;

    sw_t stopwatch;

    const auto add_active_set = [&](auto ss_idx) {
        if (!screen_is_active[ss_idx]) {
            screen_is_active[ss_idx] = true;
            active_set.push_back(ss_idx);
        }
    };

    while (1) {
        stopwatch.start();
        solve_active(
            state, 
            check_user_interrupt
        );
        active_time += stopwatch.elapsed();

        check_user_interrupt();
        ++iters;
        value_t convg_measure;
        stopwatch.start();
        coordinate_descent(
            state,
            util::counting_iterator<size_t>(0),
            util::counting_iterator<size_t>(screen_set.size()),
            convg_measure,
            add_active_set
        );
        screen_time += stopwatch.elapsed();

        if (convg_measure < tol) break;
        if (iters >= max_iters) {
            throw util::adelie_core_solver_error(
                "Maximum number of iterations reached."
            );
        }
    }
}

} // namespace pin

template <class StateType>
ADELIE_CORE_STRONG_INLINE 
void update_screen_derived(
    StateType& state
)
{
    using state_t = std::decay_t<StateType>;
    using value_t = typename state_t::value_t;
    using colmat_value_t = typename state_t::colmat_value_t;
    using rowarr_value_t = typename state_t::rowarr_value_t;

    const auto& quad_c = state.quad_c;
    const auto& quad_d = state.quad_d;
    const auto& quad_alpha = state.quad_alpha;
    const auto& quad_Sigma = state.quad_Sigma;
    const auto& screen_set = state.screen_set;
    auto& A = *state.A;
    auto& screen_hashset = state.screen_hashset;
    auto& screen_beta = state.screen_beta;
    auto& screen_is_active = state.screen_is_active;
    auto& screen_vars = state.screen_vars;
    auto& screen_ASigma = state.screen_ASigma;
    auto& screen_grad = state.screen_grad;

    /* update screen_hashset */
    {
        const auto old_screen_size = screen_hashset.size();
        const auto screen_set_new_begin = std::next(screen_set.begin(), old_screen_size);
        screen_hashset.insert(screen_set_new_begin, screen_set.end());
    }

    /* update screen_beta */
    screen_beta.resize(screen_set.size(), 0);

    /* update screen_is_active */
    screen_is_active.resize(screen_set.size(), false);

    /* update screen_ASigma */
    /* update screen_vars */
    const auto old_screen_size = screen_vars.size();
    const auto d = quad_Sigma.cols();
    screen_ASigma.resize(d * screen_set.size());
    Eigen::Map<rowarr_value_t> screen_ASigma_map(
        screen_ASigma.data(),
        screen_set.size(),
        d
    );
    screen_vars.resize(screen_set.size()); 
    for (int i = old_screen_size; i < screen_set.size(); ++i) {
        const auto k = screen_set[i];
        const auto quad_alpha_k = quad_alpha[k];
        auto AkSigma = screen_ASigma_map.row(i);
        A.rmul(k, quad_Sigma, AkSigma);
        const Eigen::Map<colmat_value_t> AkSigma_map(
            AkSigma.data(), AkSigma.size(), 1
        );
        util::rowvec_type<value_t, 1> var;
        A.rmul(k, AkSigma, var);
        screen_vars[i] = quad_c * (quad_d * quad_alpha_k * quad_alpha_k + var[0]);
    }

    /* update screen_grad */
    screen_grad.resize(screen_set.size());
}

template <class StateType>
ADELIE_CORE_STRONG_INLINE 
void screen(
    StateType& state
)
{
    const auto screen_rule = state.screen_rule;
    const auto& screen_hashset = state.screen_hashset;
    const auto& grad = state.grad;
    auto& screen_set = state.screen_set;

    const auto is_screen = [&](auto i) {
        return screen_hashset.find(i) != screen_hashset.end();
    };

    if (screen_rule == util::nnqp_screen_rule_type::_greedy) {
        for (int i = 0; i < grad.size(); ++i) {
            if (is_screen(i)) continue;
            if (grad[i] > 0) screen_set.push_back(i);
        }
    } else {
        throw util::adelie_core_solver_error(
            "Unknown screen rule!"
        );
    }

    update_screen_derived(state);
}

template <class StateType,
          class CUIType=util::no_op>
ADELIE_CORE_STRONG_INLINE
void fit(
    StateType& state,
    CUIType check_user_interrupt = CUIType()
)
{
    pin::solve(state, check_user_interrupt);
}

template <class StateType>
ADELIE_CORE_STRONG_INLINE
void update_invariance(StateType& state)
{
    using state_t = std::decay_t<StateType>;
    using vec_value_t = typename state_t::vec_value_t;
    const auto& quad_c = state.quad_c;
    const auto& quad_d = state.quad_d;
    const auto& quad_alpha = state.quad_alpha;
    const auto& quad_Sigma = state.quad_Sigma;
    const auto& linear_v = state.linear_v;
    const auto& resid_alpha = state.resid_alpha;
    const auto& resid_A = state.resid_A;
    auto& A = *state.A;
    auto& grad = state.grad;

    vec_value_t Sigma_beta(resid_A.size());
    Sigma_beta.matrix().noalias() = quad_c * (resid_A.matrix() * quad_Sigma);
    A.mul(Sigma_beta, grad);
    grad = linear_v - ((quad_c * quad_d * resid_alpha) * quad_alpha + grad);
}

template <class StateType>
ADELIE_CORE_STRONG_INLINE
bool kkt(StateType& state)
{
    const auto& screen_hashset = state.screen_hashset;
    const auto& grad = state.grad;

    const auto is_screen = [&](auto i) {
        return screen_hashset.find(i) != screen_hashset.end();
    };

    for (int k = 0; k < grad.size(); ++k) {
        if (is_screen(k)) continue;
        const auto grad_k = grad[k];
        if (grad_k > 0) return false;
    }

    return true;
}

template <class StateType,
          class CUIType>
inline void solve(
    StateType&& state,
    CUIType check_user_interrupt
)
{
    using sw_t = util::Stopwatch;

    const auto active_set = state.active_set;
    auto& benchmark_screen = state.benchmark_screen;
    auto& benchmark_invariance = state.benchmark_invariance;
    auto& benchmark_kkt = state.benchmark_kkt;
    
    sw_t sw;

    sw.start();
    update_invariance(state);
    benchmark_invariance += sw.elapsed();

    while (1) {
        // ==================================================================================== 
        // Screening step
        // ==================================================================================== 
        sw.start();
        screen(state);
        benchmark_screen += sw.elapsed();

        // ==================================================================================== 
        // Fit step
        // ==================================================================================== 
        fit(state, check_user_interrupt);

        // ==================================================================================== 
        // Invariance step
        // ==================================================================================== 
        sw.start();
        update_invariance(state);
        benchmark_invariance += sw.elapsed();

        // ==================================================================================== 
        // KKT step
        // ==================================================================================== 
        sw.start();
        const bool kkt_passed = kkt(state);
        benchmark_kkt += sw.elapsed();

        if (kkt_passed) break;
    } // end while(1)
}

} // namespace nnqp_low_rank
} // namespace optimization
} // namespace adelie_core