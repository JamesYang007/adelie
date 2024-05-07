#pragma once
#include <adelie_core/util/stopwatch.hpp>
#include <adelie_core/util/types.hpp>

namespace adelie_core {
namespace optimization {

template <class MatrixType,
          class ValueType=typename std::decay_t<MatrixType>::value_t,
          class IndexType=Eigen::Index,
          class BoolType=bool,
          class SafeBoolType=int8_t
        >
struct StateNNQP
{
    using matrix_t = MatrixType;
    using value_t = ValueType;
    using index_t = IndexType;
    //using bool_t = BoolType;
    using safe_bool_t = SafeBoolType;
    using uset_index_t = std::unordered_set<index_t>;
    using vec_value_t = util::rowvec_type<value_t>;
    //using vec_index_t = util::rowvec_type<index_t>;
    //using vec_bool_t = util::rowvec_type<bool_t>;
    //using vec_safe_bool_t = util::rowvec_type<safe_bool_t>;
    using colarr_value_t = util::colarr_type<value_t>;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;
    //using map_cvec_index_t = Eigen::Map<const vec_index_t>;
    using map_ccolarr_value_t = Eigen::Map<const colarr_value_t>;
    using dyn_vec_value_t = std::vector<value_t>;
    using dyn_vec_index_t = std::vector<index_t>;
    using dyn_vec_bool_t = std::vector<safe_bool_t>;

    /* static states */
    const value_t quad_c;
    const value_t quad_d;
    const map_cvec_value_t quad_alpha;
    const map_ccolarr_value_t quad_Q;
    const map_cvec_value_t quad_D;

    /* configurations */
    const size_t max_iters;
    const value_t tol;

    /* dynamic states */
    matrix_t* A; 
    uset_index_t screen_hashset;
    dyn_vec_index_t screen_set; 
    dyn_vec_index_t screen_begins;
    dyn_vec_value_t screen_beta;
    dyn_vec_bool_t screen_is_active;
    dyn_vec_index_t active_set;
    vec_value_t grad;
    //vec_value_t abs_grad;

    /* benchmark */
    double time_solve = 0;

    StateNNQP(
        matrix_t& A,
        value_t quad_c,
        value_t quad_d,
        const Eigen::Ref<const vec_value_t>& quad_alpha,
        const Eigen::Ref<const colarr_value_t>& quad_Q,
        const Eigen::Ref<const vec_value_t>& quad_D,
        size_t max_iters,
        value_t tol
    ):
        quad_c(quad_c),
        quad_d(quad_d),
        quad_alpha(quad_alpha.data(), quad_alpha.size()),
        quad_Q(quad_Q.data(), quad_Q.rows(), quad_Q.cols()),
        quad_D(quad_D.data(), quad_D.rows(), quad_D.cols()),
        max_iters(max_iters),
        tol(tol),
        A(&A),
        grad(A->rows())
    {}

    void reset()
    {
        screen_hashset.clear();
        screen_set.clear();
        screen_begins.clear();
        screen_beta.clear();
        screen_is_active.clear();
        active_set.clear();
        grad.setZero();
    }


    void coordinate_descent()
    {
    }

    void solve(
        const Eigen::Ref<const vec_value_t>& v
    )
    {
        using sw_t = util::Stopwatch;

    }
};


} // namespace optimization
} // namespace adelie_core