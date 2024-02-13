#include "decl.hpp"
#include <adelie_core/optimization/search_pivot.hpp>
#include <adelie_core/optimization/symmetric_penalty.hpp>

namespace py = pybind11;
namespace ad = adelie_core;

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
}