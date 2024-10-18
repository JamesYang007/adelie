#include "py_decl.hpp"
#include <adelie_core/matrix/matrix_constraint_dense.hpp>
#include <adelie_core/optimization/linqp_full.hpp>
#include <adelie_core/optimization/lasso_full.hpp>
#include <adelie_core/optimization/nnqp_full.hpp>
#include <adelie_core/optimization/pinball_full.hpp>
#include <adelie_core/optimization/search_pivot.hpp>
#include <adelie_core/optimization/symmetric_penalty.hpp>
#include <adelie_core/util/stopwatch.hpp>

namespace py = pybind11;
namespace ad = adelie_core;
using namespace pybind11::literals; // to bring in the `_a` literal

py::tuple search_pivot(
    const Eigen::Ref<const ad::util::rowvec_type<double>>& x,
    const Eigen::Ref<const ad::util::rowvec_type<double>>& y
)
{
    ad::util::rowvec_type<double> mses(x.size());
    const auto idx = ad::optimization::search_pivot(x, y, mses);
    return py::make_tuple(idx, mses);
}

double symmetric_penalty(
    const Eigen::Ref<const ad::util::rowvec_type<double>>& x,
    double alpha
)
{
    return ad::optimization::symmetric_penalty(x, alpha);
}

template <class MatrixType>
void nnqp_full(py::module_& m, const char* name)
{
    using state_t = ad::optimization::StateNNQPFull<MatrixType>;
    using matrix_t = typename state_t::matrix_t;
    using value_t = typename state_t::value_t;
    using vec_value_t = typename state_t::vec_value_t;
    py::class_<state_t>(m, name, R"delimiter(
    Solves the non-negative quadratic program (NNQP).

    The non-negative quadratic program is given by

    .. math::
        \begin{align*}
            \mathrm{minimize}_{x \geq 0} 
            \frac{1}{2} x^\top Q x - v^\top x
        \end{align*}

    where :math:`Q` is a dense positive semi-definite matrix.

    Parameters
    ----------
    quad : (n, n) ndarray
        Full positive semi-definite dense matrix :math:`Q`.
    max_iters : int
        Maximum number of coordinate descent iterations.
    tol : float
        Convergence tolerance.
    x : (n,) ndarray
        Solution vector.
    grad : (n,) ndarray
        Gradient vector :math:`v - Q x`.
    )delimiter")
        .def(py::init<
            const Eigen::Ref<const matrix_t>&,
            size_t,
            value_t,
            Eigen::Ref<vec_value_t>,
            Eigen::Ref<vec_value_t> 
        >(),
            py::arg("quad").noconvert(),
            py::arg("max_iters"),
            py::arg("tol"),
            py::arg("x"),
            py::arg("grad")
        )
        .def_readonly("quad", &state_t::quad)
        .def_readonly("max_iters", &state_t::max_iters)
        .def_readonly("tol", &state_t::tol)
        .def_readonly("iters", &state_t::iters)
        .def_readonly("x", &state_t::x)
        .def_readonly("grad", &state_t::grad)
        .def_readonly("time_elapsed", &state_t::time_elapsed)
        .def("solve", [](state_t& state) {
            using sw_t = ad::util::Stopwatch;
            sw_t sw;
            sw.start();
            state.solve();
            state.time_elapsed = sw.elapsed();
        })
        ;
}

template <class MatrixType>
void lasso_full(py::module_& m, const char* name)
{
    using state_t = ad::optimization::StateLassoFull<MatrixType>;
    using matrix_t = typename state_t::matrix_t;
    using value_t = typename state_t::value_t;
    using vec_value_t = typename state_t::vec_value_t;
    py::class_<state_t>(m, name, R"delimiter(
    Solves the lasso problem.

    The lasso is given by

    .. math::
        \begin{align*}
            \mathrm{minimize} 
            \frac{1}{2} x^\top Q x - v^\top x + \omega^\top \abs{x}
        \end{align*}

    where :math:`Q` is a dense positive semi-definite matrix
    and :math:`\omega` is a non-negative vector.

    Parameters
    ----------
    quad : (n, n) ndarray
        Full positive semi-definite dense matrix :math:`Q`.
    penalty : (n,) ndarray
        Penalty factor :math:`\omega`.
    max_iters : int
        Maximum number of coordinate descent iterations.
    tol : float
        Convergence tolerance.
    x : (n,) ndarray
        Solution vector.
    grad : (n,) ndarray
        Gradient vector :math:`v - Q x`.
    )delimiter")
        .def(py::init<
            const Eigen::Ref<const matrix_t>&,
            const Eigen::Ref<const vec_value_t>&,
            size_t,
            value_t,
            Eigen::Ref<vec_value_t>,
            Eigen::Ref<vec_value_t> 
        >(),
            py::arg("quad").noconvert(),
            py::arg("penalty").noconvert(),
            py::arg("max_iters"),
            py::arg("tol"),
            py::arg("x"),
            py::arg("grad")
        )
        .def_readonly("quad", &state_t::quad)
        .def_readonly("penalty", &state_t::penalty)
        .def_readonly("max_iters", &state_t::max_iters)
        .def_readonly("tol", &state_t::tol)
        .def_readonly("iters", &state_t::iters)
        .def_readonly("x", &state_t::x)
        .def_readonly("grad", &state_t::grad)
        .def_readonly("time_elapsed", &state_t::time_elapsed)
        .def("solve", [](state_t& state) {
            using sw_t = ad::util::Stopwatch;
            sw_t sw;
            sw.start();
            state.solve();
            state.time_elapsed = sw.elapsed();
        })
        ;
}

template <class MatrixType>
void pinball_full(py::module_& m, const char* name)
{
    using state_t = ad::optimization::StatePinballFull<MatrixType>;
    using matrix_t = typename state_t::matrix_t;
    using value_t = typename state_t::value_t;
    using vec_value_t = typename state_t::vec_value_t;
    py::class_<state_t>(m, name, R"delimiter(
    Solves the pinball least squares problem.

    The pinball least squares problem is given by

    .. math::
        \begin{align*}
            \mathrm{minimize} 
            \frac{1}{2} x^\top Q x - v^\top x + \omega_+^\top x_+ + \omega_-^\top x_-
        \end{align*}

    where :math:`Q` is a dense positive semi-definite matrix
    and :math:`\omega_{\pm}` are non-negative vectors.

    Parameters
    ----------
    quad : (n, n) ndarray
        Full positive semi-definite dense matrix :math:`Q`.
    penalty_neg : (n,) ndarray
        Penalty factor :math:`\omega_-` on the non-positive values.
    penalty_pos : (n,) ndarray
        Penalty factor :math:`\omega_+` on the non-negative values.
    y_var : float
        Scale of the problem.
    max_iters : int
        Maximum number of coordinate descent iterations.
    tol : float
        Convergence tolerance.
    x : (n,) ndarray
        Solution vector.
    grad : (n,) ndarray
        Gradient vector :math:`v - Q x`.
    )delimiter")
        .def(py::init<
            const Eigen::Ref<const matrix_t>&,
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_value_t>&,
            value_t,
            size_t,
            value_t,
            Eigen::Ref<vec_value_t>,
            Eigen::Ref<vec_value_t> 
        >(),
            py::arg("quad").noconvert(),
            py::arg("penalty_neg").noconvert(),
            py::arg("penalty_pos").noconvert(),
            py::arg("y_var"),
            py::arg("max_iters"),
            py::arg("tol"),
            py::arg("x"),
            py::arg("grad")
        )
        .def_readonly("quad", &state_t::quad)
        .def_readonly("penalty_neg", &state_t::penalty_neg)
        .def_readonly("penalty_pos", &state_t::penalty_pos)
        .def_readonly("y_var", &state_t::y_var)
        .def_readonly("max_iters", &state_t::max_iters)
        .def_readonly("tol", &state_t::tol)
        .def_readonly("iters", &state_t::iters)
        .def_readonly("x", &state_t::x)
        .def_readonly("grad", &state_t::grad)
        .def_readonly("time_elapsed", &state_t::time_elapsed)
        .def("solve", [](state_t& state) {
            using sw_t = ad::util::Stopwatch;
            sw_t sw;
            sw.start();
            state.solve();
            state.time_elapsed = sw.elapsed();
        })
        ;
}

template <class MatrixType>
void linqp_full(py::module_& m, const char* name)
{
    using state_t = ad::optimization::StateLinQPFull<MatrixType>;
    using matrix_t = typename state_t::matrix_t;
    using value_t = typename state_t::value_t;
    using vec_value_t = typename state_t::vec_value_t;
    py::class_<state_t>(m, name, R"delimiter(
    Solves the QP problem with linear inequality constraints.

    The QP problem with linear inequality constraints is given by

    .. math::
        \begin{align*}
            \mathrm{minimize} \quad&
            \frac{1}{2} x^\top Q x - v^\top x 
            \\\text{subject to} \quad&
            -\ell \leq Ax \leq u
        \end{align*}

    where :math:`Q` is a dense positive semi-definite matrix.

    Parameters
    ----------
    quad : (n, n) ndarray
        Full positive semi-definite dense matrix :math:`Q`.
    linear : (n,) ndarray
        Linear term :math:`v`.
    A : (m, n) ndarray
        Constraint matrix :math:`A`.
    lower : (n,) ndarray
        Lower bound :math:`\ell`.
    upper : (n,) ndarray
        Upper bound :math:`u`.
    max_iters : int
        Maximum number of Newton iterations.
    tol : float
        Convergence tolerance.
    slack : float
        Backtracking slackness to ensure strict feasibility.
    x : (n,) ndarray
        Solution vector.
    )delimiter")
        .def(py::init<
            const Eigen::Ref<const matrix_t>&,
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const matrix_t>&,
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_value_t>&,
            size_t,
            value_t,
            value_t,
            value_t,
            value_t,
            value_t,
            size_t,
            Eigen::Ref<vec_value_t> 
        >(),
            py::arg("quad").noconvert(),
            py::arg("linear").noconvert(),
            py::arg("A").noconvert(),
            py::arg("lower").noconvert(),
            py::arg("upper").noconvert(),
            py::arg("max_iters"),
            py::arg("relaxed_tol"),
            py::arg("tol"),
            py::arg("slack"),
            py::arg("lmda_max"),
            py::arg("lmda_min"),
            py::arg("lmda_path_size"),
            py::arg("x")
        )
        .def_readonly("quad", &state_t::quad)
        .def_readonly("linear", &state_t::linear)
        .def_readonly("A", &state_t::A)
        .def_readonly("lower", &state_t::lower)
        .def_readonly("upper", &state_t::upper)
        .def_readonly("max_iters", &state_t::max_iters)
        .def_readonly("relaxed_tol", &state_t::relaxed_tol)
        .def_readonly("tol", &state_t::tol)
        .def_readonly("slack", &state_t::slack)
        .def_readonly("iters", &state_t::iters)
        .def_readonly("backtrack_iters", &state_t::backtrack_iters)
        .def_readonly("lmda_max", &state_t::lmda_max)
        .def_readonly("lmda_min", &state_t::lmda_min)
        .def_readonly("lmda_path_size", &state_t::lmda_path_size)
        .def_readonly("x", &state_t::x)
        .def_readonly("time_elapsed", &state_t::time_elapsed)
        .def_property_readonly("buffer_size", &state_t::buffer_size)
        .def("solve", [](
            state_t& state, 
            Eigen::Ref<vec_value_t> buff
        ) {
            using sw_t = ad::util::Stopwatch;
            sw_t sw;
            sw.start();
            state.solve(buff);
            state.time_elapsed = sw.elapsed();
        })
        ;
}

void register_optimization(py::module_& m)
{
    m.def("search_pivot", &search_pivot, R"delimiter(
    Searches for a pivot point given a sequence of 2-D points.

    This function assumes that :math:`y` is generated from the linear model
    :math:`y = \beta_0 + \beta_1 (p - x) 1(x \leq p) + \epsilon`
    where `\epsilon \sim (0, \sigma^2)` and :math:`p` is some pivot point.

    Parameters
    ----------
    x : (n,) ndarray
        Sorted in increasing order of the independent variable.
    y : (n,) ndarray
        Corresponding response vector.
    
    Returns
    -------
    (idx, mses) : tuple
        ``idx`` is the index at which the minimum MSE occurs (i.e. the estimated pivot index)
        and ``mses`` is the list of MSEs computed by making index ``i`` the choice of the pivot.
    )delimiter");

    m.def("symmetric_penalty", &symmetric_penalty, R"delimiter(
    Solves the minimization of the elastic net penalty along the ones vector.

    The symmetric penalty optimization problem is given by

    .. math::
        \begin{align*}
            \mathrm{minimize}_{t} \sum\limits_{i=1}^K \left(
                \frac{1-\alpha}{2} (a_i - t)^2 + \alpha |a_i-t|
            \right)
        \end{align*}

    where :math:`a` is a fixed vector sorted in increasing order
    and :math:`\alpha \in [0,1]`.

    Parameters
    ----------
    x : (K,) ndarray
        Increasing sequence of values. 
    alpha : float
        Elastic net penalty.

    Returns
    -------
    t_star : float
        The argmin of the minimization problem.
    )delimiter");

    linqp_full<ad::util::colmat_type<double>>(m, "StateLinQPFull");
    nnqp_full<ad::util::colmat_type<double>>(m, "StateNNQPFull");
    lasso_full<ad::util::colmat_type<double>>(m, "StateLassoFull");
    pinball_full<ad::util::colmat_type<double>>(m, "StatePinballFull");
}