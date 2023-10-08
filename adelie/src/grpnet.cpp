#include "decl.hpp"
#include <adelie_core/matrix/matrix_pin_cov_base.hpp>
#include <adelie_core/matrix/matrix_pin_naive_base.hpp>
#include <adelie_core/state/state_pin_cov.hpp>
#include <adelie_core/state/state_pin_naive.hpp>
#include <adelie_core/grpnet/solve_base.hpp>
#include <adelie_core/grpnet/solve_pin_cov.hpp>
#include <adelie_core/grpnet/solve_pin_naive.hpp>

namespace py = pybind11;
namespace ad = adelie_core;
using namespace pybind11::literals; // to bring in the `_a` literal

// =================================================================
// Helper functions
// =================================================================
py::tuple transform_data(
    ad::util::rowmat_type<double> X,
    const Eigen::Ref<const ad::util::rowvec_type<int>>& groups,
    const Eigen::Ref<const ad::util::rowvec_type<int>>& group_sizes,
    size_t n_threads
) 
{
    ad::util::rowvec_type<double> d(X.cols());
    ad::grpnet::transform_data(
        X, groups, group_sizes, n_threads, d
    );
    return py::make_tuple(X, d);
}

double objective(
    const Eigen::Ref<const ad::util::rowvec_type<double>>& beta,
    const Eigen::Ref<const ad::util::rowmat_type<double>>& X,
    const Eigen::Ref<const ad::util::rowvec_type<double>>& y,
    const Eigen::Ref<const ad::util::rowvec_type<int>>& groups,
    const Eigen::Ref<const ad::util::rowvec_type<int>>& group_sizes,
    double lmda,
    double alpha,
    const Eigen::Ref<const ad::util::rowvec_type<double>>& penalty
)
{
    return ad::grpnet::objective(
        beta, X, y, groups, group_sizes, lmda, alpha, penalty
    );
}

// =================================================================
// Solve Pinned Method
// =================================================================

template <class StateType>
py::dict solve_pin_naive(StateType state)
{
    const auto update_coefficients_f = [](
        const auto& L,
        const auto& v,
        auto l1,
        auto l2,
        auto tol,
        size_t max_iters,
        auto& x,
        auto& iters,
        auto& buffer1,
        auto& buffer2
    ){
        ad::grpnet::update_coefficients(
            L, v, l1, l2, tol, max_iters, x, iters, buffer1, buffer2
        );
    };

    std::string error;
    try {
        ad::grpnet::solve_pin_naive(state, update_coefficients_f);
    } catch(const std::exception& e) {
        error = e.what(); 
    }

    return py::dict("state"_a=state, "error"_a=error);
} 

template <class StateType>
py::dict solve_pin_cov(StateType state)
{
    const auto update_coefficients_f = [](
        const auto& L,
        const auto& v,
        auto l1,
        auto l2,
        auto tol,
        size_t max_iters,
        auto& x,
        auto& iters,
        auto& buffer1,
        auto& buffer2
    ){
        ad::grpnet::update_coefficients(
            L, v, l1, l2, tol, max_iters, x, iters, buffer1, buffer2
        );
    };

    std::string error;
    try {
        ad::grpnet::solve_pin_cov(state, update_coefficients_f);
    } catch(const std::exception& e) {
        error = e.what(); 
    }

    return py::dict("state"_a=state, "error"_a=error);
} 

// =================================================================
// Solve Method
// =================================================================

double lambda_max(
    const Eigen::Ref<const ad::util::rowvec_type<double>>& abs_grad,
    double alpha,
    const Eigen::Ref<const ad::util::rowvec_type<double>>& penalty
)
{
    return ad::grpnet::lambda_max(abs_grad, alpha, penalty);
}

auto create_lambdas(
    size_t n_lambdas,
    double min_ratio,
    double lmda_max
)
{
    ad::util::rowvec_type<double> lmdas(n_lambdas);
    ad::grpnet::create_lambdas(n_lambdas, min_ratio, lmda_max, lmdas);
    return lmdas;
}

template <class T> 
using state_pin_naive_t = ad::state::StatePinNaive<ad::matrix::MatrixPinNaiveBase<T>>;
template <class T> 
using state_pin_cov_t = ad::state::StatePinCov<ad::matrix::MatrixPinCovBase<T>>;

void register_grpnet(py::module_& m)
{
    /* helpers */
    m.def("objective", &objective);
    m.def("transform_data", &transform_data, R"delimiter(
    Transforms data by rotation via SVD.

    Each block :math:`X_k` is transformed into :math:`U_k D_k` where
    :math:`X_k = U_k D_k V_k^\top` is the singular-value decomposition.

    Parameters
    ----------
    X : (n, p) np.ndarray
        Feature matrix.
    groups : (G,) np.ndarray
        List of starting indices to each group where `G` is the number of groups.
        ``groups[i]`` is the starting index of the ``i`` th group. 
    group_sizes : (G,) np.ndarray
        List of group sizes corresponding to each element in ``groups``.
        ``group_sizes[i]`` is the group size of the ``i`` th group. 
    n_threads : int
        Number of threads.

    Returns
    -------
    X, d : np.ndarray, np.ndarray
        ``X`` is the newly transformed data matrix.
        ``d`` is the list of squared singular values of ``X``.
    )delimiter");

    /* solve pinned method */
    m.def("solve_pin_naive_64", &solve_pin_naive<state_pin_naive_t<double>>);
    m.def("solve_pin_naive_32", &solve_pin_naive<state_pin_naive_t<float>>);
    m.def("solve_pin_cov_64", &solve_pin_cov<state_pin_cov_t<double>>);
    m.def("solve_pin_cov_32", &solve_pin_cov<state_pin_cov_t<float>>);

    /* solve method */
    m.def("lambda_max", &lambda_max);
    m.def("create_lambdas", &create_lambdas);
}