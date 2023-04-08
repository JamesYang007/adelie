#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Core>
#include <objective.hpp>

namespace py = pybind11;
using namespace pybind11::literals; // to bring in the `_a` literal

static double compute_h_min(
    const Eigen::Ref<Eigen::VectorXd>& vbuffer1,
    const Eigen::Ref<Eigen::VectorXd>& v,
    double l1
) 
{
    return glstudy::compute_h_min(vbuffer1, v, l1); 
}

static py::dict compute_h_max(
    const Eigen::Ref<Eigen::VectorXd>& vbuffer1,
    const Eigen::Ref<Eigen::VectorXd>& v,
    double zero_tol=1e-10
)
{
    const auto out = glstudy::compute_h_max(vbuffer1, v, zero_tol); 
    py::dict d("h_max"_a=std::get<0>(out), "vbuffer1_min_nzn"_a=std::get<1>(out));
    return d;
}

static double block_norm_objective(
    double h,
    const Eigen::Ref<Eigen::VectorXd>& D,
    const Eigen::Ref<Eigen::VectorXd>& v,
    double l1
)
{
    return glstudy::block_norm_objective(h, D, v, l1);
}

static double objective_data(
    const Eigen::Ref<Eigen::MatrixXd>& X,
    const Eigen::Ref<Eigen::VectorXd>& y,
    const Eigen::Ref<Eigen::VectorXi>& groups,
    const Eigen::Ref<Eigen::VectorXi>& group_sizes,
    double alpha,
    const Eigen::Ref<Eigen::VectorXd>& penalty,
    double lmda,
    const Eigen::Ref<Eigen::VectorXd>& beta)
{
    return glstudy::objective_data(X, y, groups, group_sizes, alpha, penalty, lmda, beta);
}