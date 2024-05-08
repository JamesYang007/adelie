#include "decl.hpp"
#include <adelie_core/optimization/nnls.hpp>
#include <adelie_core/optimization/nnqp.hpp>
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
    const auto d = quad.cols();
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
    const auto d = X.cols();
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

struct MatrixConstraintDense
{
    using value_t = double;
    using dense_t = ad::util::rowarr_type<value_t>;
    using vec_value_t = ad::util::rowvec_type<value_t>;
    using colmat_value_t = ad::util::colmat_type<value_t>;

    const Eigen::Map<const dense_t> dense;

    MatrixConstraintDense(
        const Eigen::Ref<const dense_t>& dense
    ):
        dense(dense.data(), dense.rows(), dense.cols()) 
    {}

    void rmul(
        int i, 
        const Eigen::Ref<const colmat_value_t>& S, 
        Eigen::Ref<vec_value_t> out
    )
    {
        out.matrix().noalias() = dense.row(i).matrix() * S.transpose();
    }

    void rtmul(
        int i,
        value_t s,
        Eigen::Ref<vec_value_t> out
    )
    {
        out += s * dense.row(i);
    }

    void mul(
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
    )
    {
        out.matrix().noalias() = v.matrix() * dense.matrix().transpose(); 
    }

    int rows() const { return dense.rows(); }
    int cols() const { return dense.cols();}
};

void matrix_constraint_dense(py::module_& m)
{
    using matrix_t = MatrixConstraintDense; 
    using dense_t = typename matrix_t::dense_t;
    py::class_<matrix_t>(m, "MatrixConstraintDense64C")
        .def(py::init<
            const Eigen::Ref<const dense_t>&
        >(),
            py::arg("dense").noconvert()
        )
        .def("rmul", &matrix_t::rmul)
        .def("rtmul", &matrix_t::rtmul)
        .def("mul", &matrix_t::mul)
        .def("rows", &matrix_t::rows)
        .def("cols", &matrix_t::cols)
        ;
}

template <class MatrixType>
void nnqp(py::module_& m)
{
    using state_t = ad::optimization::StateNNQP<MatrixType>;
    using matrix_t = typename state_t::matrix_t;
    using value_t = typename state_t::value_t;
    using vec_value_t = typename state_t::vec_value_t;
    using colmat_value_t = typename state_t::colmat_value_t;
    using rowarr_value_t = typename state_t::rowarr_value_t;
    py::class_<state_t>(m, "StateNNQP")
        .def(py::init<
            matrix_t&,
            value_t,
            value_t,
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const colmat_value_t>&,
            const Eigen::Ref<const vec_value_t>&,
            size_t,
            value_t,
            const std::string& 
        >(),
            py::arg("A"),
            py::arg("quad_c"),
            py::arg("quad_d"),
            py::arg("quad_alpha"),
            py::arg("quad_Sigma"),
            py::arg("linear_v"),
            py::arg("max_iters"),
            py::arg("tol"),
            py::arg("screen_rule")
        )
        .def_readonly("quad_c", &state_t::quad_c)
        .def_readonly("quad_d", &state_t::quad_d)
        .def_readonly("quad_alpha", &state_t::quad_alpha)
        .def_readonly("quad_Sigma", &state_t::quad_Sigma)
        .def_readonly("linear_v", &state_t::linear_v)
        .def_readonly("max_iters", &state_t::max_iters)
        .def_readonly("tol", &state_t::tol)
        .def_readonly("iters", &state_t::iters)
        .def_readonly("A", &state_t::A)
        .def_readonly("screen_hashset", &state_t::screen_hashset)
        .def_readonly("screen_set", &state_t::screen_set)
        .def_readonly("screen_beta", &state_t::screen_beta)
        .def_readonly("screen_is_active", &state_t::screen_is_active)
        .def_readonly("screen_vars", &state_t::screen_vars)
        .def_property_readonly("screen_ASigma", [](const state_t& state) {
            return Eigen::Map<const rowarr_value_t>(
                state.screen_ASigma.data(),
                state.screen_set.size(),
                state.quad_Sigma.cols()
            );
        })
        .def_readonly("active_set", &state_t::active_set)
        .def_readonly("resid_A", &state_t::resid_A)
        .def_readonly("resid_alpha", &state_t::resid_alpha)
        .def_readonly("grad", &state_t::grad)
        .def_readonly("benchmark_screen", &state_t::benchmark_screen)
        .def_readonly("benchmark_invariance", &state_t::benchmark_invariance)
        .def_readonly("benchmark_fit_screen", &state_t::benchmark_fit_screen)
        .def_readonly("benchmark_fit_active", &state_t::benchmark_fit_active)
        .def_readonly("benchmark_kkt", &state_t::benchmark_kkt)
        .def("solve", [](state_t& state) {
            const auto check_user_interrupt = [&]() {
                if (PyErr_CheckSignals() != 0) {
                    throw py::error_already_set();
                }
            };
            ad::optimization::nnqp::solve(state, check_user_interrupt);
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

    matrix_constraint_dense(m);
    nnqp<MatrixConstraintDense>(m);
}