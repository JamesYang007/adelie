#pragma once
#include <adelie_core/util/functional.hpp>
#include <adelie_core/util/types.hpp>

#ifndef ADELIE_CORE_STATE_PINBALL_TP
#define ADELIE_CORE_STATE_PINBALL_TP \
    template <\
        class MatrixType,\
        class ValueType,\
        class IndexType,\
        class BoolType\
    >
#endif
#ifndef ADELIE_CORE_STATE_PINBALL
#define ADELIE_CORE_STATE_PINBALL \
    StatePinball<\
        MatrixType,\
        ValueType,\
        IndexType,\
        BoolType\
    >
#endif

namespace adelie_core {
namespace state {

template <
    class MatrixType, 
    class ValueType=typename std::decay_t<MatrixType>::value_t,
    class IndexType=Eigen::Index,
    class BoolType=bool
>
class StatePinball
{
public:
    using matrix_t = MatrixType;
    using value_t = ValueType;
    using index_t = IndexType;
    using bool_t = BoolType;
    using vec_value_t = util::rowvec_type<value_t>;
    using vec_index_t = util::rowvec_type<index_t>;
    using vec_bool_t = util::rowvec_type<bool_t>;
    using colmat_value_t = util::colmat_type<value_t>;
    using rowmat_value_t = util::rowmat_type<value_t>;
    using map_vec_value_t = Eigen::Map<vec_value_t>;
    using map_vec_index_t = Eigen::Map<vec_index_t>;
    using map_vec_bool_t = Eigen::Map<vec_bool_t>;
    using map_rowmat_value_t = Eigen::Map<rowmat_value_t>;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;
    using map_ccolmat_value_t = Eigen::Map<const colmat_value_t>;

    matrix_t* A;

    /* static states */
    const value_t y_var;
    const map_ccolmat_value_t S;
    const map_cvec_value_t penalty_neg;
    const map_cvec_value_t penalty_pos;

    /* configurations */
    const size_t kappa;
    const size_t max_iters;
    const value_t tol;

    /* dynamic states */
    size_t screen_set_size;
    map_vec_index_t screen_set;
    map_vec_bool_t is_screen;
    map_vec_value_t screen_ASAT_diag;
    map_rowmat_value_t screen_AS;
    size_t active_set_size;
    map_vec_index_t active_set;
    map_vec_bool_t is_active;
    map_vec_value_t beta;
    map_vec_value_t resid;
    map_vec_value_t grad;
    value_t loss;
    size_t iters = 0;
    size_t n_kkt = 0;

    // debug
    std::vector<double> benchmark_fit_active;
    std::vector<double> benchmark_fit_screen;
    std::vector<double> benchmark_gradient;
    std::vector<double> benchmark_viols_sort;

    std::vector<vec_value_t> dbg_beta;
    std::vector<vec_index_t> dbg_active_set;
    std::vector<size_t> dbg_iter;
    std::vector<value_t> dbg_loss;

private:
    void initialize();

public:
    virtual ~StatePinball() {};

    explicit StatePinball(
        matrix_t& A,
        value_t y_var,
        const Eigen::Ref<const colmat_value_t>& S,
        const Eigen::Ref<const vec_value_t>& penalty_neg,
        const Eigen::Ref<const vec_value_t>& penalty_pos,
        size_t kappa,
        size_t max_iters,
        value_t tol,
        size_t screen_set_size,
        Eigen::Ref<vec_index_t> screen_set,
        Eigen::Ref<vec_bool_t> is_screen,
        Eigen::Ref<vec_value_t> screen_ASAT_diag,
        Eigen::Ref<rowmat_value_t> screen_AS,
        size_t active_set_size,
        Eigen::Ref<vec_index_t> active_set,
        Eigen::Ref<vec_bool_t> is_active,
        Eigen::Ref<vec_value_t> beta,
        Eigen::Ref<vec_value_t> resid,
        Eigen::Ref<vec_value_t> grad,
        value_t loss
    ):
        A(&A),
        y_var(y_var),
        S(S.data(), S.rows(), S.cols()),
        penalty_neg(penalty_neg.data(), penalty_neg.size()),
        penalty_pos(penalty_pos.data(), penalty_pos.size()),
        kappa(kappa),
        max_iters(max_iters),
        tol(tol),
        screen_set_size(screen_set_size),
        screen_set(screen_set.data(), screen_set.size()),
        is_screen(is_screen.data(), is_screen.size()),
        screen_ASAT_diag(screen_ASAT_diag.data(), screen_ASAT_diag.size()),
        screen_AS(screen_AS.data(), screen_AS.rows(), screen_AS.cols()),
        active_set_size(active_set_size),
        active_set(active_set.data(), active_set.size()),
        is_active(is_active.data(), is_active.size()),
        beta(beta.data(), beta.size()),
        resid(resid.data(), resid.size()),
        grad(grad.data(), grad.size()),
        loss(loss)
    {
        initialize();
    }

    void solve(
        std::function<void()> check_user_interrupt =util::no_op()
    );
};

} // namespace state
} // namespace adelie_core