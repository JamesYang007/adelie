#include "decl.hpp"
#include <adelie_core/bcd/constrained/admm.hpp>
#include <adelie_core/bcd/constrained/coordinate_descent.hpp>
#include <adelie_core/bcd/constrained/proximal_newton.hpp>
#include <adelie_core/bcd/unconstrained/brent.hpp>
#include <adelie_core/bcd/unconstrained/ista.hpp>
#include <adelie_core/bcd/unconstrained/newton.hpp>
#include <adelie_core/util/stopwatch.hpp>

namespace py = pybind11;
namespace ad = adelie_core;
using namespace pybind11::literals; // to bring in the `_a` literal

// ================================================================
// Utility functions
// ================================================================

double root_lower_bound(
    const Eigen::Ref<const ad::util::rowvec_type<double>>& quad,
    const Eigen::Ref<const ad::util::rowvec_type<double>>& linear,
    double l1
) 
{
    return ad::bcd::root_lower_bound(quad, linear, l1); 
}

double root_upper_bound(
    const Eigen::Ref<const ad::util::rowvec_type<double>>& quad,
    const Eigen::Ref<const ad::util::rowvec_type<double>>& linear,
    double zero_tol=1e-10
)
{
    const auto out = ad::bcd::root_upper_bound(quad, linear, zero_tol); 
    return std::get<0>(out);
}

double root_function(
    double h,
    const Eigen::Ref<const ad::util::rowvec_type<double>>& D,
    const Eigen::Ref<const ad::util::rowvec_type<double>>& v,
    double l1
)
{
    return ad::bcd::root_function(h, D, v, l1);
}

// ================================================================
// Unconstrained
// ================================================================

py::dict unconstrained_ista_solver(
    const Eigen::Ref<const ad::util::rowvec_type<double>>& L,
    const Eigen::Ref<const ad::util::rowvec_type<double>>& v,
    double l1,
    double l2,
    double tol,
    size_t max_iters
)
{
    ad::util::rowvec_type<double> x(L.size());
    size_t iters = 0;
    ad::bcd::unconstrained::ista_solver(L, v, l1, l2, tol, max_iters, x, iters);
    py::dict d("beta"_a=x, "iters"_a=iters);
    return d;
}

py::dict unconstrained_fista_solver(
    const Eigen::Ref<const ad::util::rowvec_type<double>>& L,
    const Eigen::Ref<const ad::util::rowvec_type<double>>& v,
    double l1,
    double l2,
    double tol,
    size_t max_iters
)
{
    ad::util::rowvec_type<double> x(L.size());
    size_t iters = 0;
    ad::bcd::unconstrained::fista_solver(L, v, l1, l2, tol, max_iters, x, iters);
    py::dict d("beta"_a=x, "iters"_a=iters);
    return d;
}

py::dict unconstrained_fista_adares_solver(
    const Eigen::Ref<const ad::util::rowvec_type<double>>& L,
    const Eigen::Ref<const ad::util::rowvec_type<double>>& v,
    double l1,
    double l2,
    double tol,
    size_t max_iters
)
{
    ad::util::rowvec_type<double> x(L.size());
    size_t iters = 0;
    ad::bcd::unconstrained::fista_adares_solver(L, v, l1, l2, tol, max_iters, x, iters);
    py::dict d("beta"_a=x, "iters"_a=iters);
    return d;
}

py::dict unconstrained_newton_solver(
    const Eigen::Ref<const ad::util::rowvec_type<double>>& L,
    const Eigen::Ref<const ad::util::rowvec_type<double>>& v,
    double l1,
    double l2,
    double tol,
    size_t max_iters
)
{
    ad::util::rowvec_type<double> x(L.size());
    ad::util::rowvec_type<double> buffer1(L.size());
    ad::util::rowvec_type<double> buffer2(L.size());
    size_t iters = 0;
    ad::bcd::unconstrained::newton_solver(L, v, l1, l2, tol, max_iters, x, iters, buffer1, buffer2);

    py::dict d("beta"_a=x, "iters"_a=iters);
    return d;
}

py::dict unconstrained_newton_brent_solver(
    const Eigen::Ref<const ad::util::rowvec_type<double>>& L,
    const Eigen::Ref<const ad::util::rowvec_type<double>>& v,
    double l1,
    double l2,
    double tol,
    size_t max_iters
)
{
    ad::util::rowvec_type<double> x(L.size());
    ad::util::rowvec_type<double> buffer1(L.size());
    ad::util::rowvec_type<double> buffer2(L.size());
    size_t iters = 0;
    ad::bcd::unconstrained::newton_brent_solver(L, v, l1, l2, tol, tol, max_iters, x, iters, buffer1, buffer2);

    py::dict d("beta"_a=x, "iters"_a=iters);
    return d;
}

py::dict unconstrained_newton_abs_solver(
    const Eigen::Ref<const ad::util::rowvec_type<double>>& L,
    const Eigen::Ref<const ad::util::rowvec_type<double>>& v,
    double l1,
    double l2,
    double tol,
    size_t max_iters
)
{
    ad::util::rowvec_type<double> x(L.size());
    ad::util::rowvec_type<double> buffer1(L.size());
    ad::util::rowvec_type<double> buffer2(L.size());
    size_t iters = 0;
    ad::bcd::unconstrained::newton_abs_solver(L, v, l1, l2, tol, max_iters, x, iters, buffer1, buffer2);

    py::dict d("beta"_a=x, "iters"_a=iters);
    return d;
}

py::dict unconstrained_newton_abs_debug_solver(
    const Eigen::Ref<const ad::util::rowvec_type<double>>& L,
    const Eigen::Ref<const ad::util::rowvec_type<double>>& v,
    double l1,
    double l2,
    double tol,
    size_t max_iters,
    bool smart_init
)
{
    double h_min, h_max;
    ad::util::rowvec_type<double> x(L.size());
    ad::util::rowvec_type<double> buffer1(L.size());
    ad::util::rowvec_type<double> buffer2(L.size());
    std::vector<double> iters;
    iters.reserve(L.size());
    std::vector<double> smart_iters;
    smart_iters.reserve(L.size());
    ad::bcd::unconstrained::newton_abs_debug_solver(
        L, v, l1, l2, tol, max_iters, smart_init, 
        h_min, h_max, x, iters, smart_iters, buffer1, buffer2
    );
    
    py::dict d(
        "beta"_a=x,
        "h_min"_a=h_min,
        "h_max"_a=h_max,
        "iters"_a=iters,
        "smart_iters"_a=smart_iters
    );
    return d;
}

py::dict unconstrained_brent_solver(
    const Eigen::Ref<const ad::util::rowvec_type<double>>& L,
    const Eigen::Ref<const ad::util::rowvec_type<double>>& v,
    double l1,
    double l2,
    double tol,
    size_t max_iters
)
{
    ad::util::rowvec_type<double> x(v.size());
    size_t iters = 0;
    ad::bcd::unconstrained::brent_solver(L, v, l1, l2, tol, max_iters, x, iters);
    py::dict d("beta"_a=x, "iters"_a=iters);
    return d;
}

// ================================================================
// Constrained
// ================================================================

py::dict constrained_admm_solver(
    const Eigen::Ref<const ad::util::rowvec_type<double>>& quad_c,
    double l1,
    double l2,
    const Eigen::Ref<const ad::util::rowmat_type<double>>& Q_c,
    const Eigen::Ref<const ad::util::rowmat_type<double>>& AQ_c,
    const Eigen::Ref<const ad::util::rowvec_type<double>>& QTv_c,
    const Eigen::Ref<const ad::util::rowmat_type<double>>& A,
    const Eigen::Ref<const ad::util::rowvec_type<double>>& b,
    double rho,
    size_t max_iters,
    double tol_abs,
    double tol_rel
)
{
    using sw_t = ad::util::Stopwatch;

    const auto m = A.rows();
    const auto d = A.cols();
    ad::util::rowvec_type<double> x(d); x.setZero();
    ad::util::rowvec_type<double> z(m); z.setZero();
    ad::util::rowvec_type<double> u(m); u.setZero();
    ad::util::rowvec_type<double> buff(3*m + 4*d);
    size_t iters;
    sw_t sw;
    sw.start();
    ad::bcd::constrained::admm_solver(
        quad_c, l1, l2, Q_c, AQ_c, QTv_c, A, b, rho, max_iters, tol_abs, tol_rel,
        x, z, u, iters, buff
    );
    const auto elapsed = sw.elapsed();
    py::dict dct("x"_a=x, "z"_a=z, "u"_a=u, "iters"_a=iters, "time_elapsed"_a=elapsed);
    return dct;
}

py::dict constrained_coordinate_descent_solver(
    const Eigen::Ref<const ad::util::rowvec_type<double>>& mu0,
    const Eigen::Ref<const ad::util::rowvec_type<double>>& quad,
    const Eigen::Ref<const ad::util::rowvec_type<double>>& linear,
    double l1,
    double l2,
    const Eigen::Ref<const ad::util::rowmat_type<double>>& A,
    const Eigen::Ref<const ad::util::rowvec_type<double>>& b,
    size_t max_iters,
    double tol,
    size_t pnewton_max_iters,
    double pnewton_tol,
    size_t newton_max_iters,
    double newton_tol
)
{
    using sw_t = ad::util::Stopwatch;

    const auto m = A.rows();
    const auto d = A.cols();
    ad::util::rowvec_type<double> A_vars = A.array().square().rowwise().sum();
    ad::util::rowvec_type<double> buff(4*d);
    ad::util::rowvec_type<double> x(d);
    ad::util::rowvec_type<double> mu = mu0;
    ad::util::rowvec_type<double> mu_resid = (
        linear.matrix() - mu.matrix() * A
    );
    double mu_rsq = mu_resid.square().sum();
    size_t iters;
    sw_t sw;
    sw.start();
    ad::bcd::constrained::coordinate_descent_solver(
        quad, linear, l1, l2, A, b, A_vars,
        max_iters, tol, pnewton_max_iters, pnewton_tol, newton_max_iters, newton_tol,
        iters, x, mu, mu_resid, mu_rsq, buff
    );
    const auto elapsed = sw.elapsed();
    py::dict dct("x"_a=x, "mu"_a=mu, "iters"_a=iters, "time_elapsed"_a=elapsed);
    return dct;
}

py::dict constrained_proximal_newton_solver(
    const Eigen::Ref<const ad::util::rowvec_type<double>>& mu0,
    const Eigen::Ref<const ad::util::rowvec_type<double>>& quad,
    const Eigen::Ref<const ad::util::rowvec_type<double>>& linear,
    double l1,
    double l2,
    const Eigen::Ref<const ad::util::rowmat_type<double>>& A,
    const Eigen::Ref<const ad::util::rowvec_type<double>>& b,
    size_t max_iters,
    double tol,
    size_t newton_max_iters,
    double newton_tol,
    size_t nnls_max_iters,
    double nnls_tol,
    double nnls_dtol
)
{
    using sw_t = ad::util::Stopwatch;

    const auto m = A.rows();
    const auto d = A.cols();

    size_t iters;
    ad::util::rowvec_type<double> buff(m*(m+4+d)+3*d);
    ad::util::rowvec_type<double> x(d);
    ad::util::rowvec_type<double> mu = mu0;
    ad::util::rowvec_type<double> mu_resid = (
        linear.matrix() - x.matrix() * A
    );
    ad::util::rowvec_type<double> AT_vars = (
        A.array().square().rowwise().sum()
    );
    ad::util::rowmat_type<double> AAT = A * A.transpose();

    sw_t sw;
    sw.start();
    ad::bcd::constrained::proximal_newton_solver(
        quad, linear, l1, l2, A, b, AT_vars, AAT,
        max_iters, tol, newton_max_iters, newton_tol, nnls_max_iters, nnls_tol, nnls_dtol,
        iters, x, mu, mu_resid, buff
    );
    const auto time_elapsed = sw.elapsed();

    return py::dict(
        "x"_a=x, "mu"_a=mu, "iters"_a=iters, "time_elapsed"_a=time_elapsed
    );
}

void register_bcd(py::module_& m)
{
    /* utility functions */
    m.def("root_function", &root_function);
    m.def("root_lower_bound", &root_lower_bound);
    m.def("root_upper_bound", &root_upper_bound);

    /* constrained */
    m.def("constrained_admm_solver", &constrained_admm_solver);
    m.def("constrained_coordinate_descent_solver", &constrained_coordinate_descent_solver);
    m.def("constrained_proximal_newton_solver", &constrained_proximal_newton_solver);

    /* unconstrained */
    m.def("unconstrained_brent_solver", &unconstrained_brent_solver);
    m.def("unconstrained_fista_adares_solver", &unconstrained_fista_adares_solver);
    m.def("unconstrained_fista_solver", &unconstrained_fista_solver);
    m.def("unconstrained_ista_solver", &unconstrained_ista_solver);
    m.def("unconstrained_newton_abs_debug_solver", &unconstrained_newton_abs_debug_solver);
    m.def("unconstrained_newton_abs_solver", &unconstrained_newton_abs_solver);
    m.def("unconstrained_newton_brent_solver", &unconstrained_newton_brent_solver);
    m.def("unconstrained_newton_solver", &unconstrained_newton_solver);
}