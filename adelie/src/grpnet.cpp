#include "decl.hpp"
//#include <adelie_core/optimization/grpelnet_pin_cov.hpp>
//#include <adelie_core/matrix/cov_cache.hpp>
#include <adelie_core/matrix/matrix_base.hpp>
#include <adelie_core/state/pin_naive.hpp>
#include <adelie_core/grpnet/solve_pin_naive.hpp>

namespace py = pybind11;
namespace ad = adelie_core;
using namespace pybind11::literals; // to bring in the `_a` literal

double objective(
    const Eigen::Ref<ad::util::rowvec_type<double>>& beta,
    const Eigen::Ref<ad::util::rowmat_type<double>>& X,
    const Eigen::Ref<ad::util::rowvec_type<double>>& y,
    const Eigen::Ref<ad::util::rowvec_type<int>>& groups,
    const Eigen::Ref<ad::util::rowvec_type<int>>& group_sizes,
    double lmda,
    double alpha,
    const Eigen::Ref<ad::util::rowvec_type<double>>& penalty
)
{
    return ad::grpnet::objective(
        beta, X, y, groups, group_sizes, lmda, alpha, penalty
    );
}

// =================================================================
// Solve Pinned Method
// =================================================================

template <class StateType>
py::dict solve_pin_naive(StateType state)
{
    const auto update_coefficients_f = [](
        const auto& L,
        const auto& v,
        auto l1,
        auto l2,
        auto tol,
        size_t max_iters,
        auto& x,
        auto& iters,
        auto& buffer1,
        auto& buffer2
    ){
        ad::grpnet::update_coefficients(
            L, v, l1, l2, tol, max_iters, x, iters, buffer1, buffer2
        );
    };

    std::string error;
    try {
        ad::grpnet::solve_pin_naive(state, update_coefficients_f);
    } catch(const std::exception& e) {
        error = e.what(); 
    }

    return py::dict("state"_a=state, "error"_a=error);
} 

template <class T> 
using pin_naive_t = ad::state::PinNaive<ad::matrix::MatrixBase<T>>;

void register_grpnet(py::module_& m)
{
    m.def("objective", &objective);
    m.def("solve_pin_naive_64", &solve_pin_naive<pin_naive_t<double>>);
    m.def("solve_pin_naive_32", &solve_pin_naive<pin_naive_t<float>>);
}