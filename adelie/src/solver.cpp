#include "decl.hpp"
#include <adelie_core/matrix/matrix_cov_base.hpp>
#include <adelie_core/matrix/matrix_naive_base.hpp>
#include <adelie_core/state/state_basil_naive.hpp>
#include <adelie_core/state/state_pin_cov.hpp>
#include <adelie_core/state/state_pin_naive.hpp>
#include <adelie_core/solver/solve_basil_base.hpp>
#include <adelie_core/solver/solve_basil_naive.hpp>
#include <adelie_core/solver/solve_pin_cov.hpp>
#include <adelie_core/solver/solve_pin_naive.hpp>

namespace py = pybind11;
namespace ad = adelie_core;
using namespace pybind11::literals; // to bring in the `_a` literal

// =================================================================
// Helper functions
// =================================================================
double objective(
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
    return ad::solver::objective(
        beta0, beta, X, y, groups, group_sizes, lmda, alpha, penalty, weights
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
        ad::solver::update_coefficients(
            L, v, l1, l2, tol, max_iters, x, iters, buffer1, buffer2
        );
    };

    const auto check_user_interrupt = [&](auto) {
        if (PyErr_CheckSignals() != 0) {
            throw py::error_already_set();
        }
    };

    std::string error;
    try {
        ad::solver::naive::solve_pin(state, update_coefficients_f, check_user_interrupt);
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
        ad::solver::update_coefficients(
            L, v, l1, l2, tol, max_iters, x, iters, buffer1, buffer2
        );
    };

    const auto check_user_interrupt = [&](auto) {
        if (PyErr_CheckSignals() != 0) {
            throw py::error_already_set();
        }
    };

    std::string error;
    try {
        ad::solver::cov::solve_pin(state, update_coefficients_f, check_user_interrupt);
    } catch(const std::exception& e) {
        error = e.what(); 
    }

    return py::dict("state"_a=state, "error"_a=error);
} 

// =================================================================
// Solve Basil Method
// =================================================================

template <class StateType>
py::dict solve_basil_naive(StateType state)
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
        ad::solver::update_coefficients(
            L, v, l1, l2, tol, max_iters, x, iters, buffer1, buffer2
        );
    };

    const auto check_user_interrupt = [&](auto) {
        if (PyErr_CheckSignals() != 0) {
            throw py::error_already_set();
        }
    };

    std::string error;
    try {
        ad::solver::naive::solve_basil(state, update_coefficients_f, check_user_interrupt);
    } catch(const std::exception& e) {
        error = e.what(); 
    }

    return py::dict("state"_a=state, "error"_a=error);
} 

template <class T> 
using state_pin_naive_t = ad::state::StatePinNaive<ad::matrix::MatrixNaiveBase<T>>;
template <class T> 
using state_pin_cov_t = ad::state::StatePinCov<ad::matrix::MatrixCovBase<T>>;
template <class T> 
using state_basil_naive_t = ad::state::StateBasilNaive<ad::matrix::MatrixNaiveBase<T>>;

void register_solver(py::module_& m)
{
    /* helpers */
    m.def("objective", &objective);

    /* solve pinned method */
    m.def("solve_pin_naive_64", &solve_pin_naive<state_pin_naive_t<double>>);
    m.def("solve_pin_naive_32", &solve_pin_naive<state_pin_naive_t<float>>);
    m.def("solve_pin_cov_64", &solve_pin_cov<state_pin_cov_t<double>>);
    m.def("solve_pin_cov_32", &solve_pin_cov<state_pin_cov_t<float>>);

    /* solve basil method */
    m.def("solve_basil_naive_64", &solve_basil_naive<state_basil_naive_t<double>>);
    m.def("solve_basil_naive_32", &solve_basil_naive<state_basil_naive_t<float>>);
}