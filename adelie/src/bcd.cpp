#include "decl.hpp"
#include <adelie_core/bcd.hpp>
#include <adelie_core/util/stopwatch.hpp>

namespace py = pybind11;
namespace ad = adelie_core;
using namespace pybind11::literals; // to bring in the `_a` literal

py::dict ista_solver(
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
    ad::bcd::ista_solver(L, v, l1, l2, tol, max_iters, x, iters);
    py::dict d("beta"_a=x, "iters"_a=iters);
    return d;
}

py::dict fista_solver(
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
    ad::bcd::fista_solver(L, v, l1, l2, tol, max_iters, x, iters);
    py::dict d("beta"_a=x, "iters"_a=iters);
    return d;
}

py::dict fista_adares_solver(
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
    ad::bcd::fista_adares_solver(L, v, l1, l2, tol, max_iters, x, iters);
    py::dict d("beta"_a=x, "iters"_a=iters);
    return d;
}

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

py::dict newton_solver(
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
    ad::bcd::newton_solver(L, v, l1, l2, tol, max_iters, x, iters, buffer1, buffer2);

    py::dict d("beta"_a=x, "iters"_a=iters);
    return d;
}

py::dict newton_brent_solver(
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
    ad::bcd::newton_brent_solver(L, v, l1, l2, tol, tol, max_iters, x, iters, buffer1, buffer2);

    py::dict d("beta"_a=x, "iters"_a=iters);
    return d;
}

py::dict newton_abs_solver(
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
    ad::bcd::newton_abs_solver(L, v, l1, l2, tol, max_iters, x, iters, buffer1, buffer2);

    py::dict d("beta"_a=x, "iters"_a=iters);
    return d;
}

py::dict newton_abs_debug_solver(
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
    ad::bcd::newton_abs_debug_solver(
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

py::dict brent_solver(
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
    ad::bcd::brent_solver(L, v, l1, l2, tol, max_iters, x, iters);
    py::dict d("beta"_a=x, "iters"_a=iters);
    return d;
}

py::dict admm_cnstr_solver(
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
    ad::bcd::admm_cnstr_solver(
        quad_c, l1, l2, Q_c, AQ_c, QTv_c, A, b, rho, max_iters, tol_abs, tol_rel,
        x, z, u, iters, buff
    );
    const auto elapsed = sw.elapsed();
    py::dict dct("x"_a=x, "z"_a=z, "u"_a=u, "iters"_a=iters, "time_elapsed"_a=elapsed);
    return dct;
}

void register_bcd(py::module_& m)
{
    m.def("admm_cnstr_solver", &admm_cnstr_solver);
    m.def("brent_solver", &brent_solver);
    m.def("brent_solver", &brent_solver);
    m.def("ista_solver", &ista_solver);
    m.def("fista_adares_solver", &fista_adares_solver);
    m.def("fista_solver", &fista_solver);
    m.def("newton_solver", &newton_solver);
    m.def("newton_brent_solver", &newton_brent_solver);
    m.def("newton_abs_solver", &newton_abs_solver);
    m.def("newton_abs_debug_solver", &newton_abs_debug_solver);
    m.def("root_function", &root_function);
    m.def("root_lower_bound", &root_lower_bound);
    m.def("root_upper_bound", &root_upper_bound);
}