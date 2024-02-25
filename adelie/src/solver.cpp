#include "decl.hpp"
#include <adelie_core/glm/glm_base.hpp>
#include <adelie_core/glm/glm_multibase.hpp>
#include <adelie_core/matrix/matrix_cov_base.hpp>
#include <adelie_core/matrix/matrix_naive_base.hpp>
#include <adelie_core/state/state_gaussian_cov.hpp>
#include <adelie_core/state/state_gaussian_naive.hpp>
#include <adelie_core/state/state_gaussian_pin_cov.hpp>
#include <adelie_core/state/state_gaussian_pin_naive.hpp>
#include <adelie_core/state/state_glm_naive.hpp>
#include <adelie_core/state/state_multigaussian_naive.hpp>
#include <adelie_core/state/state_multiglm_naive.hpp>
#include <adelie_core/solver/solver_gaussian_cov.hpp>
#include <adelie_core/solver/solver_gaussian_naive.hpp>
#include <adelie_core/solver/solver_gaussian_pin_cov.hpp>
#include <adelie_core/solver/solver_gaussian_pin_naive.hpp>
#include <adelie_core/solver/solver_gaussian_pin_naive.hpp>
#include <adelie_core/solver/solver_glm_naive.hpp>
#include <adelie_core/solver/solver_multigaussian_naive.hpp>
#include <adelie_core/solver/solver_multiglm_naive.hpp>

namespace py = pybind11;
namespace ad = adelie_core;
using namespace pybind11::literals; // to bring in the `_a` literal

// =================================================================
// Helper functions
// =================================================================
template <class T>
ad::util::rowvec_type<T> compute_penalty_sparse(
    const Eigen::Ref<const ad::util::rowvec_type<int>>& groups,
    const Eigen::Ref<const ad::util::rowvec_type<int>>& group_sizes,
    const Eigen::Ref<const ad::util::rowvec_type<T>>& penalty,
    T alpha,
    const Eigen::SparseMatrix<T, Eigen::RowMajor>& betas,
    size_t n_threads
)
{
    using value_t = T;
    using vec_value_t = ad::util::rowvec_type<value_t>;
    using sp_mat_value_t = Eigen::SparseMatrix<T, Eigen::RowMajor>;

    vec_value_t out(betas.outerSize());

    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (int k = 0; k < betas.outerSize(); ++k)
    {
        typename sp_mat_value_t::InnerIterator it(betas, k);
        value_t pnlty = 0;
        for (int i = 0; i < groups.size(); ++i) {
            if (!it) break;
            const auto g = groups[i];
            const auto gs = group_sizes[i];
            const auto pg = penalty[i];
            value_t norm = 0;
            while (it && (it.index() >= g) && (it.index() < g + gs)) {
                norm += it.value() * it.value();
                ++it;
            }
            norm = std::sqrt(norm);
            pnlty += pg * norm * (alpha + 0.5 * (1-alpha) * norm);
        }
        out[k] = pnlty;
    }

    return out;
}

template <class T>
ad::util::rowvec_type<T> compute_penalty_dense(
    const Eigen::Ref<const ad::util::rowvec_type<int>>& groups,
    const Eigen::Ref<const ad::util::rowvec_type<int>>& group_sizes,
    const Eigen::Ref<const ad::util::rowvec_type<T>>& penalty,
    T alpha,
    const Eigen::Ref<const ad::util::rowmat_type<T>>& betas,
    size_t n_threads
)
{
    using value_t = T;
    using vec_value_t = ad::util::rowvec_type<value_t>;

    vec_value_t out(betas.rows());

    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (int k = 0; k < betas.rows(); ++k)
    {
        const auto beta_k = betas.row(k);
        value_t pnlty = 0;
        for (int i = 0; i < groups.size(); ++i) {
            const auto g = groups[i];
            const auto gs = group_sizes[i];
            const auto pg = penalty[i];
            value_t norm = Eigen::Map<const vec_value_t>(
                beta_k.data() + g, gs
            ).matrix().norm();
            pnlty += pg * norm * (alpha + 0.5 * (1-alpha) * norm);
        }
        out[k] = pnlty;
    }

    return out;
}

// =================================================================
// Solve Pinned Method
// =================================================================

template <class StateType>
py::dict solve_gaussian_pin_cov(StateType state)
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
    sw_t sw;
    sw.start();
    try {
        ad::solver::gaussian::pin::cov::solve(
            state, update_coefficients_f, check_user_interrupt
        );
    } catch(const std::exception& e) {
        error = e.what(); 
    }
    double total_time = sw.elapsed();

    return py::dict("state"_a=state, "error"_a=error, "total_time"_a=total_time);
} 

template <class StateType>
py::dict solve_gaussian_pin_naive(StateType state)
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
    sw_t sw;
    sw.start();
    try {
        ad::solver::gaussian::pin::naive::solve(
            state, update_coefficients_f, check_user_interrupt
        );
    } catch(const std::exception& e) {
        error = e.what(); 
    }
    double total_time = sw.elapsed();

    return py::dict("state"_a=state, "error"_a=error, "total_time"_a=total_time);
} 

// =================================================================
// Solve Gaussian Method
// =================================================================

template <class StateType>
py::dict solve_gaussian_cov(
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
        ad::solver::gaussian::cov::solve(
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

template <class StateType, class GlmType>
py::dict solve_glm_naive(
    StateType state,
    GlmType& glm,
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
            state, glm, display_progress_bar, 
            update_coefficients_f, check_user_interrupt
        );
    } catch(const std::exception& e) {
        error = e.what(); 
    }

    double total_time = sw.elapsed();
    return py::dict("state"_a=state, "error"_a=error, "total_time"_a=total_time);
} 

template <class StateType, class GlmType>
py::dict solve_multiglm_naive(
    StateType state,
    GlmType& glm,
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
        ad::solver::multiglm::naive::solve(
            state, glm, display_progress_bar, 
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
using state_gaussian_cov_t = ad::state::StateGaussianCov<ad::matrix::MatrixCovBase<T>>;
template <class T> 
using state_gaussian_naive_t = ad::state::StateGaussianNaive<ad::matrix::MatrixNaiveBase<T>>;
template <class T> 
using state_multigaussian_naive_t = ad::state::StateMultiGaussianNaive<ad::matrix::MatrixNaiveBase<T>>;
template <class T> 
using state_glm_naive_t = ad::state::StateGlmNaive<ad::matrix::MatrixNaiveBase<T>>;
template <class T> 
using state_multiglm_naive_t = ad::state::StateMultiGlmNaive<ad::matrix::MatrixNaiveBase<T>>;
template <class T>
using glm_t = ad::glm::GlmBase<T>;
template <class T>
using glm_multi_t = ad::glm::GlmMultiBase<T>;

void register_solver(py::module_& m)
{
    /* helper functions */
    m.def("compute_penalty_sparse", &compute_penalty_sparse<double>);
    m.def("compute_penalty_dense", &compute_penalty_dense<double>);

    /* solve pinned method */
    m.def("solve_gaussian_pin_naive_64", &solve_gaussian_pin_naive<state_gaussian_pin_naive_t<double>>);
    m.def("solve_gaussian_pin_naive_32", &solve_gaussian_pin_naive<state_gaussian_pin_naive_t<float>>);
    m.def("solve_gaussian_pin_cov_64", &solve_gaussian_pin_cov<state_gaussian_pin_cov_t<double>>);
    m.def("solve_gaussian_pin_cov_32", &solve_gaussian_pin_cov<state_gaussian_pin_cov_t<float>>);

    /* solve gaussian method */
    m.def("solve_gaussian_cov_64", &solve_gaussian_cov<state_gaussian_cov_t<double>>);
    m.def("solve_gaussian_cov_32", &solve_gaussian_cov<state_gaussian_cov_t<float>>);
    m.def("solve_gaussian_naive_64", &solve_gaussian_naive<state_gaussian_naive_t<double>>);
    m.def("solve_gaussian_naive_32", &solve_gaussian_naive<state_gaussian_naive_t<float>>);
    m.def("solve_multigaussian_naive_64", &solve_multigaussian_naive<state_multigaussian_naive_t<double>>);
    m.def("solve_multigaussian_naive_32", &solve_multigaussian_naive<state_multigaussian_naive_t<float>>);

    /* solve glm method */
    m.def("solve_glm_naive_64", &solve_glm_naive<state_glm_naive_t<double>, glm_t<double>>);
    m.def("solve_glm_naive_32", &solve_glm_naive<state_glm_naive_t<float>, glm_t<float>>);
    m.def("solve_multiglm_naive_64", &solve_multiglm_naive<state_multiglm_naive_t<double>, glm_multi_t<double>>);
    m.def("solve_multiglm_naive_32", &solve_multiglm_naive<state_multiglm_naive_t<float>, glm_multi_t<float>>);
}