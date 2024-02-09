#include "decl.hpp"
#include <adelie_core/glm/glm_base.hpp>
#include <adelie_core/matrix/matrix_cov_base.hpp>
#include <adelie_core/matrix/matrix_naive_base.hpp>
#include <adelie_core/state/state_gaussian_naive.hpp>
#include <adelie_core/state/state_gaussian_pin_cov.hpp>
#include <adelie_core/state/state_gaussian_pin_naive.hpp>
#include <adelie_core/state/state_glm_naive.hpp>
#include <adelie_core/state/state_multigaussian_naive.hpp>
#include <adelie_core/solver/solver_gaussian_naive.hpp>
#include <adelie_core/solver/solver_gaussian_pin_cov.hpp>
#include <adelie_core/solver/solver_gaussian_pin_naive.hpp>
#include <adelie_core/solver/solver_gaussian_pin_naive.hpp>
#include <adelie_core/solver/solver_glm_naive.hpp>
#include <adelie_core/solver/solver_multigaussian_naive.hpp>

namespace py = pybind11;
namespace ad = adelie_core;
using namespace pybind11::literals; // to bring in the `_a` literal

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
py::dict solve_gaussian_naive(
    StateType state,
    bool display_progress_bar
)
{
    using sw_t = ad::util::Stopwatch;

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

    // this is to redirect std::cerr to sys.stderr in Python.
    // https://pybind11.readthedocs.io/en/stable/advanced/pycpp/utilities.html?highlight=cout#capturing-standard-output-from-ostream
    py::scoped_estream_redirect _estream;
    sw_t sw;
    sw.start();
    try {
        ad::solver::gaussian::naive::solve(
            state, display_progress_bar, 
            update_coefficients_f, check_user_interrupt
        );
    } catch(const std::exception& e) {
        error = e.what(); 
    }
    double total_time = sw.elapsed();

    return py::dict("state"_a=state, "error"_a=error, "total_time"_a=total_time);
} 

template <class StateType>
py::dict solve_multigaussian_naive(
    StateType state,
    bool display_progress_bar
)
{
    using sw_t = ad::util::Stopwatch;

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

    // this is to redirect std::cerr to sys.stderr in Python.
    // https://pybind11.readthedocs.io/en/stable/advanced/pycpp/utilities.html?highlight=cout#capturing-standard-output-from-ostream
    py::scoped_estream_redirect _estream;
    sw_t sw;
    sw.start();
    try {
        ad::solver::multigaussian::naive::solve(
            state, display_progress_bar, 
            update_coefficients_f, check_user_interrupt
        );
    } catch(const std::exception& e) {
        error = e.what(); 
    }
    double total_time = sw.elapsed();

    return py::dict("state"_a=state, "error"_a=error, "total_time"_a=total_time);
} 

// =================================================================
// Solve GLM Method
// =================================================================

template <class StateType>
py::dict solve_glm_naive(
    StateType state,
    bool display_progress_bar
)
{
    using sw_t = ad::util::Stopwatch;

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

    // this is to redirect std::cerr to sys.stderr in Python.
    // https://pybind11.readthedocs.io/en/stable/advanced/pycpp/utilities.html?highlight=cout#capturing-standard-output-from-ostream
    py::scoped_estream_redirect _estream;
    sw_t sw;
    sw.start();
    try {
        ad::solver::glm::naive::solve(
            state, display_progress_bar, 
            update_coefficients_f, check_user_interrupt
        );
    } catch(const std::exception& e) {
        error = e.what(); 
    }

    double total_time = sw.elapsed();
    return py::dict("state"_a=state, "error"_a=error, "total_time"_a=total_time);
} 

template <class T> 
using state_gaussian_pin_naive_t = ad::state::StateGaussianPinNaive<ad::matrix::MatrixNaiveBase<T>>;
template <class T> 
using state_gaussian_pin_cov_t = ad::state::StateGaussianPinCov<ad::matrix::MatrixCovBase<T>>;
template <class T> 
using state_gaussian_naive_t = ad::state::StateGaussianNaive<ad::matrix::MatrixNaiveBase<T>>;
template <class T> 
using state_multigaussian_naive_t = ad::state::StateMultiGaussianNaive<ad::matrix::MatrixNaiveBase<T>>;
template <class T> 
using state_glm_naive_t = ad::state::StateGlmNaive<
    ad::glm::GlmBase<T>,
    ad::matrix::MatrixNaiveBase<T>
>;

void register_solver(py::module_& m)
{
    /* solve pinned method */
    m.def("solve_gaussian_pin_naive_64", &solve_gaussian_pin_naive<state_gaussian_pin_naive_t<double>>);
    m.def("solve_gaussian_pin_naive_32", &solve_gaussian_pin_naive<state_gaussian_pin_naive_t<float>>);
    m.def("solve_gaussian_pin_cov_64", &solve_gaussian_pin_cov<state_gaussian_pin_cov_t<double>>);
    m.def("solve_gaussian_pin_cov_32", &solve_gaussian_pin_cov<state_gaussian_pin_cov_t<float>>);

    /* solve gaussian method */
    m.def("solve_gaussian_naive_64", &solve_gaussian_naive<state_gaussian_naive_t<double>>);
    m.def("solve_gaussian_naive_32", &solve_gaussian_naive<state_gaussian_naive_t<float>>);
    m.def("solve_multigaussian_naive_64", &solve_multigaussian_naive<state_multigaussian_naive_t<double>>);
    m.def("solve_multigaussian_naive_32", &solve_multigaussian_naive<state_multigaussian_naive_t<float>>);
    m.def("solve_glm_naive_64", &solve_glm_naive<state_glm_naive_t<double>>);
    m.def("solve_glm_naive_32", &solve_glm_naive<state_glm_naive_t<float>>);
}