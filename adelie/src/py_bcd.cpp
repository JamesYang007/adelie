#include "py_decl.hpp"
#include <adelie_core/bcd/elastic_net/unconstrained/brent.hpp>
#include <adelie_core/bcd/elastic_net/unconstrained/ista.hpp>
#include <adelie_core/bcd/elastic_net/unconstrained/newton.hpp>
#include <adelie_core/bcd/sgl/unconstrained/coordinate_descent.hpp>
#include <adelie_core/util/stopwatch.hpp>

namespace py = pybind11;
namespace ad = adelie_core;
using namespace pybind11::literals; // to bring in the `_a` literal

// ================================================================
// Utility functions
// ================================================================

double elastic_net_root_lower_bound(
    const Eigen::Ref<const ad::util::rowvec_type<double>>& quad,
    const Eigen::Ref<const ad::util::rowvec_type<double>>& linear,
    double l1
) 
{
    return ad::bcd::elastic_net::root_lower_bound(quad, linear, l1); 
}

double elastic_net_root_upper_bound(
    const Eigen::Ref<const ad::util::rowvec_type<double>>& quad,
    const Eigen::Ref<const ad::util::rowvec_type<double>>& linear,
    double l1,
    double zero_tol=1e-14
)
{
    const auto out = ad::bcd::elastic_net::root_upper_bound(quad, linear, l1, zero_tol); 
    return std::get<0>(out);
}

double elastic_net_root_function(
    double h,
    const Eigen::Ref<const ad::util::rowvec_type<double>>& D,
    const Eigen::Ref<const ad::util::rowvec_type<double>>& v,
    double l1
)
{
    return ad::bcd::elastic_net::root_function(h, D, v, l1);
}

// ================================================================
// Elastic Net Unconstrained
// ================================================================

py::dict elastic_net_unconstrained_ista_solver(
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
    ad::bcd::elastic_net::unconstrained::ista_solver(L, v, l1, l2, tol, max_iters, x, iters);
    double time = sw.elapsed();
    py::dict d("beta"_a=x, "iters"_a=iters, "time_elapsed"_a=time);
    return d;
}

py::dict elastic_net_unconstrained_fista_solver(
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
    ad::bcd::elastic_net::unconstrained::fista_solver(L, v, l1, l2, tol, max_iters, x, iters);
    double time = sw.elapsed();
    py::dict d("beta"_a=x, "iters"_a=iters, "time_elapsed"_a=time);
    return d;
}

py::dict elastic_net_unconstrained_fista_adares_solver(
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
    ad::bcd::elastic_net::unconstrained::fista_adares_solver(L, v, l1, l2, tol, max_iters, x, iters);
    double time = sw.elapsed();
    py::dict d("beta"_a=x, "iters"_a=iters, "time_elapsed"_a=time);
    return d;
}

py::dict elastic_net_unconstrained_newton_solver(
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
    ad::bcd::elastic_net::unconstrained::newton_solver(L, v, l1, l2, tol, max_iters, x, iters, buffer1, buffer2);
    double time = sw.elapsed();
    py::dict d("beta"_a=x, "iters"_a=iters, "time_elapsed"_a=time);
    return d;
}

py::dict elastic_net_unconstrained_newton_brent_solver(
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
    ad::bcd::elastic_net::unconstrained::newton_brent_solver(L, v, l1, l2, tol, tol, max_iters, x, iters, buffer1, buffer2);
    double time = sw.elapsed();
    py::dict d("beta"_a=x, "iters"_a=iters, "time_elapsed"_a=time);
    return d;
}

py::dict elastic_net_unconstrained_newton_abs_solver(
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
    ad::bcd::elastic_net::unconstrained::newton_abs_solver(L, v, l1, l2, tol, max_iters, x, iters, buffer1, buffer2);
    double time = sw.elapsed();
    py::dict d("beta"_a=x, "iters"_a=iters, "time_elapsed"_a=time);
    return d;
}

py::dict elastic_net_unconstrained_newton_abs_debug_solver(
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
    ad::bcd::elastic_net::unconstrained::newton_abs_debug_solver(
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

py::dict elastic_net_unconstrained_brent_solver(
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
    ad::bcd::elastic_net::unconstrained::brent_solver(L, v, l1, l2, tol, max_iters, x, iters);
    double time = sw.elapsed();
    py::dict d("beta"_a=x, "iters"_a=iters, "time_elapsed"_a=time);
    return d;
}

// ================================================================
// SGL Unconstrained
// ================================================================

py::dict sgl_unconstrained_coordinate_descent_solver(
    const Eigen::Ref<const ad::util::rowmat_type<double>>& S,
    const Eigen::Ref<const ad::util::rowvec_type<double>>& v,
    double l1,
    double l2,
    double tol,
    size_t max_iters
)
{
    using sw_t = ad::util::Stopwatch;
    ad::util::rowvec_type<double> x(S.cols());
    x.setZero();
    ad::util::rowvec_type<double> grad = v.matrix() - x.matrix() * S;
    size_t iters = 0;
    sw_t sw;
    sw.start();
    ad::bcd::sgl::unconstrained::coordinate_descent_solver(S, v, l1, l2, tol, max_iters, x, grad, iters);
    double time = sw.elapsed();
    py::dict d("beta"_a=x, "iters"_a=iters, "time_elapsed"_a=time);
    return d;
}

void register_bcd(py::module_& m)
{
    /* elastic net unconstrained */
    auto m_elastic_net = m.def_submodule("elastic_net", "Elastic net submodule.");
    m_elastic_net.def("root_function", &elastic_net_root_function);
    m_elastic_net.def("root_lower_bound", &elastic_net_root_lower_bound);
    m_elastic_net.def("root_upper_bound", &elastic_net_root_upper_bound);
    m_elastic_net.def("unconstrained_brent_solver", &elastic_net_unconstrained_brent_solver);
    m_elastic_net.def("unconstrained_fista_adares_solver", &elastic_net_unconstrained_fista_adares_solver);
    m_elastic_net.def("unconstrained_fista_solver", &elastic_net_unconstrained_fista_solver);
    m_elastic_net.def("unconstrained_ista_solver", &elastic_net_unconstrained_ista_solver);
    m_elastic_net.def("unconstrained_newton_abs_debug_solver", &elastic_net_unconstrained_newton_abs_debug_solver);
    m_elastic_net.def("unconstrained_newton_abs_solver", &elastic_net_unconstrained_newton_abs_solver);
    m_elastic_net.def("unconstrained_newton_brent_solver", &elastic_net_unconstrained_newton_brent_solver);
    m_elastic_net.def("unconstrained_newton_solver", &elastic_net_unconstrained_newton_solver);

    /* sgl unconstrained */
    auto m_sgl = m.def_submodule("sgl", "SGL submodule.");
    m_sgl.def("quartic_roots", &ad::bcd::sgl::quartic_roots<double>);
    m_sgl.def("root_secular", &ad::bcd::sgl::root_secular<double>);
    m_sgl.def("unconstrained_coordinate_descent_solver", &sgl_unconstrained_coordinate_descent_solver);
}