#include "py_decl.hpp"
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
    double l1,
    double zero_tol=1e-14
)
{
    const auto out = ad::bcd::root_upper_bound(quad, linear, l1, zero_tol); 
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
    using sw_t = ad::util::Stopwatch;
    ad::util::rowvec_type<double> x(L.size());
    size_t iters = 0;
    sw_t sw;
    sw.start();
    ad::bcd::unconstrained::ista_solver(L, v, l1, l2, tol, max_iters, x, iters);
    double time = sw.elapsed();
    py::dict d("beta"_a=x, "iters"_a=iters, "time_elapsed"_a=time);
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
    using sw_t = ad::util::Stopwatch;
    ad::util::rowvec_type<double> x(L.size());
    size_t iters = 0;
    sw_t sw;
    sw.start();
    ad::bcd::unconstrained::fista_solver(L, v, l1, l2, tol, max_iters, x, iters);
    double time = sw.elapsed();
    py::dict d("beta"_a=x, "iters"_a=iters, "time_elapsed"_a=time);
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
    using sw_t = ad::util::Stopwatch;
    ad::util::rowvec_type<double> x(L.size());
    size_t iters = 0;
    sw_t sw;
    sw.start();
    ad::bcd::unconstrained::fista_adares_solver(L, v, l1, l2, tol, max_iters, x, iters);
    double time = sw.elapsed();
    py::dict d("beta"_a=x, "iters"_a=iters, "time_elapsed"_a=time);
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
    using sw_t = ad::util::Stopwatch;
    ad::util::rowvec_type<double> x(L.size());
    ad::util::rowvec_type<double> buffer1(L.size());
    ad::util::rowvec_type<double> buffer2(L.size());
    size_t iters = 0;
    sw_t sw;
    sw.start();
    ad::bcd::unconstrained::newton_solver(L, v, l1, l2, tol, max_iters, x, iters, buffer1, buffer2);
    double time = sw.elapsed();
    py::dict d("beta"_a=x, "iters"_a=iters, "time_elapsed"_a=time);
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
    using sw_t = ad::util::Stopwatch;
    ad::util::rowvec_type<double> x(L.size());
    ad::util::rowvec_type<double> buffer1(L.size());
    ad::util::rowvec_type<double> buffer2(L.size());
    size_t iters = 0;
    sw_t sw;
    sw.start();
    ad::bcd::unconstrained::newton_brent_solver(L, v, l1, l2, tol, tol, max_iters, x, iters, buffer1, buffer2);
    double time = sw.elapsed();
    py::dict d("beta"_a=x, "iters"_a=iters, "time_elapsed"_a=time);
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
    using sw_t = ad::util::Stopwatch;
    ad::util::rowvec_type<double> x(L.size());
    ad::util::rowvec_type<double> buffer1(L.size());
    ad::util::rowvec_type<double> buffer2(L.size());
    size_t iters = 0;
    sw_t sw;
    sw.start();
    ad::bcd::unconstrained::newton_abs_solver(L, v, l1, l2, tol, max_iters, x, iters, buffer1, buffer2);
    double time = sw.elapsed();
    py::dict d("beta"_a=x, "iters"_a=iters, "time_elapsed"_a=time);
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
    using sw_t = ad::util::Stopwatch;
    ad::util::rowvec_type<double> x(v.size());
    size_t iters = 0;
    sw_t sw;
    sw.start();
    ad::bcd::unconstrained::brent_solver(L, v, l1, l2, tol, max_iters, x, iters);
    double time = sw.elapsed();
    py::dict d("beta"_a=x, "iters"_a=iters, "time_elapsed"_a=time);
    return d;
}

void register_bcd(py::module_& m)
{
    /* utility functions */
    m.def("root_function", &root_function);
    m.def("root_lower_bound", &root_lower_bound);
    m.def("root_upper_bound", &root_upper_bound);

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