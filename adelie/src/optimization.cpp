#include "decl.hpp"
#include <adelie_core/optimization/nnls.hpp>
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

py::dict nnls_cov_full(
    const Eigen::Ref<const ad::util::rowvec_type<double>>& x0,
    const Eigen::Ref<const ad::util::rowmat_type<double>>& quad,
    const Eigen::Ref<const ad::util::rowvec_type<double>>& linear,
    size_t max_iters,
    double tol,
    double dtol
)
{
    using sw_t = ad::util::Stopwatch;
    size_t iters;
    ad::util::rowvec_type<double> x = x0;
    ad::util::rowvec_type<double> grad = (
        linear.matrix() - x.matrix() * quad.transpose()
    );
    double loss = 0.5 * (x.matrix() * quad).dot(x.matrix()) - (linear * x).sum();
    sw_t sw;
    sw.start();
    ad::optimization::nnls_cov_full(
        quad, max_iters, tol, dtol, 
        iters, x, grad, loss, [](){return false;}
    );
    const auto time_elapsed = sw.elapsed();
    return py::dict(
        "x"_a=x, 
        "grad"_a=grad, 
        "loss"_a=loss,
        "iters"_a=iters, 
        "time_elapsed"_a=time_elapsed
    );
}

py::dict nnls_naive(
    const Eigen::Ref<const ad::util::rowvec_type<double>>& beta0,
    const Eigen::Ref<const ad::util::colmat_type<double>>& X,
    const Eigen::Ref<const ad::util::rowvec_type<double>>& y,
    size_t max_iters,
    double tol,
    double dtol
)
{
    using sw_t = ad::util::Stopwatch;
    size_t iters;
    ad::util::rowvec_type<double> X_vars = (
        X.array().square().colwise().sum()
    );
    ad::util::rowvec_type<double> beta = beta0;
    ad::util::rowvec_type<double> resid = (
        y.matrix() - beta0.matrix() * X.transpose()
    );
    double loss = 0.5 * resid.square().sum();
    sw_t sw;
    sw.start();
    ad::optimization::nnls_naive(
        X, X_vars, max_iters, tol, dtol, 
        iters, beta, resid, loss, 
        [](){return false;}, [](auto) {return false;}
    );
    const auto time_elapsed = sw.elapsed();
    return py::dict(
        "x"_a=beta, 
        "resid"_a=resid, 
        "loss"_a=loss,
        "iters"_a=iters, 
        "time_elapsed"_a=time_elapsed
    );
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
    x : (n,) np.ndarray
        Sorted in increasing order of the independent variable.
    y : (n,) np.ndarray
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
    x : (K,) np.ndarray
        Increasing sequence of values. 
    alpha : float
        Elastic net penalty.

    Returns
    -------
    t_star : float
        The argmin of the minimization problem.
    )delimiter");
    m.def("nnls_cov_full", &nnls_cov_full, R"delimiter(
    Solves the non-negative least squares (NNLS) problem
    with a full quadratic component.

    The non-negative least squares problem is given by

    .. math::
        \begin{align*}
            \mathrm{minimize}_{x \geq 0} 
            \frac{1}{2} x^\top Q x - v^\top x
        \end{align*}

    where :math:`Q` is a dense positive semi-definite matrix.

    Parameters
    ----------
    x : (d,) np.ndarray
        Initial starting value.
    quad : (d, d) np.ndarray
        Full positive semi-definite dense matrix :math:`Q`.
    linear : (d,) np.ndarray 
        Linear component :math:`v`.
    max_iters : int
        Maximum number of coordinate descent iterations.
    tol : float
        Convergence tolerance.
    dtol : float
        Difference tolerance at each coordinate update.
        If the absolute difference is below this value,
        then the update does not take place, which saves computation.
    )delimiter");
    m.def("nnls_naive", &nnls_naive, R"delimiter(
    Solves the non-negative least squares (NNLS) problem
    in the regression form.

    The non-negative least squares problem in the regression form is given by

    .. math::
        \begin{align*}
            \mathrm{minimize}_{\beta \geq 0} 
            \frac{1}{2} \|y - X\beta\|_2^2
        \end{align*}

    Parameters
    ----------
    beta0 : (d,) np.ndarray
        Initial starting value.
    X : (n, d) np.ndarray
        Feature matrix.
    y : (n,) np.ndarray
        Response vector.
    max_iters : int
        Maximum number of coordinate descent iterations.
    tol : float
        Convergence tolerance.
    dtol : float
        Difference tolerance at each coordinate update.
        If the absolute difference is below this value,
        then the update does not take place, which saves computation.
    )delimiter");
}