#include "decl.hpp"
#include <adelie_core/optimization/nnls.hpp>
#include <adelie_core/optimization/nnqp_full.hpp>
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

    value_t rmul(
        int i, 
        const Eigen::Ref<const vec_value_t>& v
    )
    {
        return (dense.row(i) * v).sum();
    }

    void rtmul(
        int i,
        value_t s,
        Eigen::Ref<vec_value_t> out
    )
    {
        out += s * dense.row(i);
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
        .def("rows", &matrix_t::rows)
        .def("cols", &matrix_t::cols)
        ;
}

// TODO: revive?
//template <class MatrixType>
//void nnqp(py::module_& m)
//{
//    using state_t = ad::optimization::StateNNQP<MatrixType>;
//    using matrix_t = typename state_t::matrix_t;
//    using value_t = typename state_t::value_t;
//    using vec_value_t = typename state_t::vec_value_t;
//    using colmat_value_t = typename state_t::colmat_value_t;
//    using rowarr_value_t = typename state_t::rowarr_value_t;
//    py::class_<state_t>(m, "StateNNQP")
//        .def(py::init<
//            matrix_t&,
//            value_t,
//            value_t,
//            const Eigen::Ref<const vec_value_t>&,
//            const Eigen::Ref<const colmat_value_t>&,
//            const Eigen::Ref<const vec_value_t>&,
//            size_t,
//            value_t,
//            const std::string& 
//        >(),
//            py::arg("A"),
//            py::arg("quad_c"),
//            py::arg("quad_d"),
//            py::arg("quad_alpha"),
//            py::arg("quad_Sigma"),
//            py::arg("linear_v"),
//            py::arg("max_iters"),
//            py::arg("tol"),
//            py::arg("screen_rule")
//        )
//        .def_readonly("quad_c", &state_t::quad_c)
//        .def_readonly("quad_d", &state_t::quad_d)
//        .def_readonly("quad_alpha", &state_t::quad_alpha)
//        .def_readonly("quad_Sigma", &state_t::quad_Sigma)
//        .def_readonly("linear_v", &state_t::linear_v)
//        .def_readonly("max_iters", &state_t::max_iters)
//        .def_readonly("tol", &state_t::tol)
//        .def_readonly("iters", &state_t::iters)
//        .def_readonly("A", &state_t::A)
//        .def_readonly("screen_hashset", &state_t::screen_hashset)
//        .def_readonly("screen_set", &state_t::screen_set)
//        .def_readonly("screen_beta", &state_t::screen_beta)
//        .def_readonly("screen_is_active", &state_t::screen_is_active)
//        .def_readonly("screen_vars", &state_t::screen_vars)
//        .def_property_readonly("screen_ASigma", [](const state_t& state) {
//            return Eigen::Map<const rowarr_value_t>(
//                state.screen_ASigma.data(),
//                state.screen_set.size(),
//                state.quad_Sigma.cols()
//            );
//        })
//        .def_readonly("active_set", &state_t::active_set)
//        .def_readonly("resid_A", &state_t::resid_A)
//        .def_readonly("resid_alpha", &state_t::resid_alpha)
//        .def_readonly("grad", &state_t::grad)
//        .def_readonly("benchmark_screen", &state_t::benchmark_screen)
//        .def_readonly("benchmark_invariance", &state_t::benchmark_invariance)
//        .def_readonly("benchmark_fit_screen", &state_t::benchmark_fit_screen)
//        .def_readonly("benchmark_fit_active", &state_t::benchmark_fit_active)
//        .def_readonly("benchmark_kkt", &state_t::benchmark_kkt)
//        .def("solve", [](state_t& state) {
//            const auto check_user_interrupt = [&]() {
//                if (PyErr_CheckSignals() != 0) {
//                    throw py::error_already_set();
//                }
//            };
//            ad::optimization::nnqp::solve(state, check_user_interrupt);
//        })
//        ;
//}

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

    //matrix_constraint_dense(m);
    //nnqp<MatrixConstraintDense>(m);
    //nnqp_basic<MatrixConstraintDense>(m);
}