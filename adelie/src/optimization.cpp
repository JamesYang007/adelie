#include "decl.hpp"
#include <adelie_core/optimization/nnls.hpp>
#include <adelie_core/optimization/nnqp_full.hpp>
#include <adelie_core/optimization/hinge_full.hpp>
#include <adelie_core/optimization/lasso_full.hpp>
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
            ad::optimization::nnqp_full(state);
            state.time_elapsed = sw.elapsed();
        })
        ;
}

template <class MatrixType>
void nnls(py::module_& m, const char* name)
{
    using state_t = ad::optimization::StateNNLS<MatrixType>;
    using matrix_t = typename state_t::matrix_t;
    using value_t = typename state_t::value_t;
    using vec_value_t = typename state_t::vec_value_t;
    py::class_<state_t>(m, name, R"delimiter(
    Solves the non-negative least squares (NNLS) problem.

    The non-negative least squares problem is given by

    .. math::
        \begin{align*}
            \mathrm{minimize}_{\beta \geq 0} 
            \frac{1}{2} \|y - X\beta\|_2^2
        \end{align*}

    Parameters
    ----------
    X : (n, d) ndarray
        Feature matrix.
    X_vars : (d,) ndarray
        :math:`\ell_2`-norm squared of the columns of ``X``.
    max_iters : int
        Maximum number of coordinate descent iterations.
    tol : float
        Convergence tolerance.
    beta : (d,) ndarray
        Solution vector.
    resid : (n,) ndarray
        Residual vector :math:`y - X \beta`.
    loss : float
        Loss :math:`1/2 \|y-X\beta\|_2^2`.
    )delimiter")
        .def(py::init<
            const Eigen::Ref<const matrix_t>&,
            const Eigen::Ref<const vec_value_t>&,
            size_t,
            value_t,
            Eigen::Ref<vec_value_t>,
            Eigen::Ref<vec_value_t>,
            value_t 
        >(),
            py::arg("X").noconvert(),
            py::arg("X_vars").noconvert(),
            py::arg("max_iters"),
            py::arg("tol"),
            py::arg("beta"),
            py::arg("resid"),
            py::arg("loss")
        )
        .def_readonly("X", &state_t::X)
        .def_readonly("X_vars", &state_t::X_vars)
        .def_readonly("max_iters", &state_t::max_iters)
        .def_readonly("tol", &state_t::tol)
        .def_readonly("beta", &state_t::beta)
        .def_readonly("resid", &state_t::resid)
        .def_readonly("loss", &state_t::loss)
        .def_readonly("time_elapsed", &state_t::time_elapsed)
        .def("solve", [](state_t& state) {
            using sw_t = ad::util::Stopwatch;
            sw_t sw;
            sw.start();
            ad::optimization::nnls(
                state, 
                [](){return false;}, 
                [](auto) {return false;}
            );
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
            ad::optimization::lasso_full(state);
            state.time_elapsed = sw.elapsed();
        })
        ;
}

template <class MatrixType>
void hinge_full(py::module_& m, const char* name)
{
    using state_t = ad::optimization::StateHingeFull<MatrixType>;
    using matrix_t = typename state_t::matrix_t;
    using value_t = typename state_t::value_t;
    using vec_value_t = typename state_t::vec_value_t;
    py::class_<state_t>(m, name, R"delimiter(
    Solves the hinge problem.

    The hinge problem is given by

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
    penalty_pos : (n,) ndarray
        Penalty factor :math:`\omega_+` on the non-negative values.
    penalty_pos : (n,) ndarray
        Penalty factor :math:`\omega_-` on the non-positive values.
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
            size_t,
            value_t,
            Eigen::Ref<vec_value_t>,
            Eigen::Ref<vec_value_t> 
        >(),
            py::arg("quad").noconvert(),
            py::arg("penalty_pos").noconvert(),
            py::arg("penalty_neg").noconvert(),
            py::arg("max_iters"),
            py::arg("tol"),
            py::arg("x"),
            py::arg("grad")
        )
        .def_readonly("quad", &state_t::quad)
        .def_readonly("penalty_pos", &state_t::penalty_pos)
        .def_readonly("penalty_neg", &state_t::penalty_neg)
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
            ad::optimization::hinge_full(state);
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

    nnqp_full<ad::util::colmat_type<double>>(m, "StateNNQPFull");
    nnls<ad::util::colmat_type<double>>(m, "StateNNLS");
    lasso_full<ad::util::colmat_type<double>>(m, "StateLassoFull");
    hinge_full<ad::util::colmat_type<double>>(m, "StateHingeFull");
}