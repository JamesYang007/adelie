#include "decl.hpp"
#include <adelie_core/matrix/matrix_cov_base.hpp>
#include <adelie_core/matrix/matrix_naive_base.hpp>
#include <adelie_core/state/state_gaussian_naive.hpp>
#include <adelie_core/state/state_gaussian_pin_cov.hpp>
#include <adelie_core/state/state_gaussian_pin_naive.hpp>
#include <adelie_core/solver/solver_gaussian_naive.hpp>
#include <adelie_core/solver/solver_gaussian_pin_cov.hpp>
#include <adelie_core/solver/solver_gaussian_pin_naive.hpp>

namespace py = pybind11;
namespace ad = adelie_core;
using namespace pybind11::literals; // to bring in the `_a` literal

// =================================================================
// Helper functions
// =================================================================
double gaussian_naive_objective(
    double beta0,
    const Eigen::Ref<const ad::util::rowvec_type<double>>& beta,
    const Eigen::Ref<const ad::util::rowmat_type<double>>& X,
    const Eigen::Ref<const ad::util::rowvec_type<double>>& y,
    const Eigen::Ref<const ad::util::rowvec_type<int>>& groups,
    const Eigen::Ref<const ad::util::rowvec_type<int>>& group_sizes,
    double lmda,
    double alpha,
    const Eigen::Ref<const ad::util::rowvec_type<double>>& penalty,
    const Eigen::Ref<const ad::util::rowvec_type<double>>& weights
)
{
    return ad::solver::gaussian::naive::objective(
        beta0, beta, X, y, groups, group_sizes, lmda, alpha, penalty, weights
    );
}

// =================================================================
// Solve Pinned Method
// =================================================================

template <class StateType>
py::dict solve_gaussian_pin_naive(StateType state)
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
        ad::solver::gaussian::pin::update_coefficients(
            L, v, l1, l2, tol, max_iters, x, iters, buffer1, buffer2
        );
    };

    const auto check_user_interrupt = [&]() {
        if (PyErr_CheckSignals() != 0) {
            throw py::error_already_set();
        }
    };

    std::string error;
    try {
        ad::solver::gaussian::pin::naive::solve(
            state, update_coefficients_f, check_user_interrupt
        );
    } catch(const std::exception& e) {
        error = e.what(); 
    }

    return py::dict("state"_a=state, "error"_a=error);
} 

template <class StateType>
py::dict solve_gaussian_pin_cov(StateType state)
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
        ad::solver::gaussian::pin::update_coefficients(
            L, v, l1, l2, tol, max_iters, x, iters, buffer1, buffer2
        );
    };

    const auto check_user_interrupt = [&]() {
        if (PyErr_CheckSignals() != 0) {
            throw py::error_already_set();
        }
    };

    std::string error;
    try {
        ad::solver::gaussian::pin::cov::solve(
            state, update_coefficients_f, check_user_interrupt
        );
    } catch(const std::exception& e) {
        error = e.what(); 
    }

    return py::dict("state"_a=state, "error"_a=error);
} 

// =================================================================
// Solve Gaussian Method
// =================================================================

template <class StateType>
py::dict solve_gaussian_naive(StateType state)
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
        ad::solver::gaussian::pin::update_coefficients(
            L, v, l1, l2, tol, max_iters, x, iters, buffer1, buffer2
        );
    };

    const auto check_user_interrupt = [&]() {
        if (PyErr_CheckSignals() != 0) {
            throw py::error_already_set();
        }
    };

    std::string error;
    try {
        ad::solver::gaussian::naive::solve(
            state, update_coefficients_f, check_user_interrupt
        );
    } catch(const std::exception& e) {
        error = e.what(); 
    }

    return py::dict("state"_a=state, "error"_a=error);
} 

template <class T> 
using state_gaussian_pin_naive_t = ad::state::StateGaussianPinNaive<ad::matrix::MatrixNaiveBase<T>>;
template <class T> 
using state_gaussian_pin_cov_t = ad::state::StateGaussianPinCov<ad::matrix::MatrixCovBase<T>>;
template <class T> 
using state_gaussian_naive_t = ad::state::StateGaussianNaive<ad::matrix::MatrixNaiveBase<T>>;

void register_solver(py::module_& m)
{
    /* helpers */
    m.def("gaussian_naive_objective", &gaussian_naive_objective);

    /* solve pinned method */
    m.def("solve_gaussian_pin_naive_64", &solve_gaussian_pin_naive<state_gaussian_pin_naive_t<double>>);
    m.def("solve_gaussian_pin_naive_32", &solve_gaussian_pin_naive<state_gaussian_pin_naive_t<float>>);
    m.def("solve_gaussian_pin_cov_64", &solve_gaussian_pin_cov<state_gaussian_pin_cov_t<double>>);
    m.def("solve_gaussian_pin_cov_32", &solve_gaussian_pin_cov<state_gaussian_pin_cov_t<float>>);

    /* solve gaussian method */
    m.def("solve_gaussian_naive_64", &solve_gaussian_naive<state_gaussian_naive_t<double>>);
    m.def("solve_gaussian_naive_32", &solve_gaussian_naive<state_gaussian_naive_t<float>>);
}