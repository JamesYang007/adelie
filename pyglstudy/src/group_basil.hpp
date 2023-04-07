#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <ghostbasil/optimization/group_basil.hpp>

namespace py = pybind11;
using namespace pybind11::literals; // to bring in the `_a` literal
                                    

template <class CheckptType>
inline py::dict convert_checkpoint(
    const CheckptType& checkpoint
)
{
    return py::dict(
        "is_initialized"_a=checkpoint.is_initialized,
        "strong_set"_a=checkpoint.strong_set,
        "strong_g1"_a=checkpoint.strong_g1,
        "strong_g2"_a=checkpoint.strong_g2,
        "strong_begins"_a=checkpoint.strong_begins,
        "strong_order"_a=checkpoint.strong_order,
        "strong_beta"_a=checkpoint.strong_beta,
        "strong_grad"_a=checkpoint.strong_grad,
        "strong_A_diag"_a=checkpoint.strong_A_diag,
        "active_set"_a=checkpoint.active_set,
        "active_g1"_a=checkpoint.active_g1,
        "active_g2"_a=checkpoint.active_g2,
        "active_begins"_a=checkpoint.active_begins,
        "active_order"_a=checkpoint.active_order,
        "is_active"_a=checkpoint.is_active,
        "grad"_a=checkpoint.grad,
        "abs_grad"_a=checkpoint.abs_grad,
        "rsq"_a=checkpoint.rsq
    );
}
                                
template <class DiagType>
inline py::dict convert_diagnostic(
    const DiagType& diag
)
{
    std::vector<py::dict> checkpoints;
    checkpoints.reserve(diag.checkpoints.size());
    for (size_t i = 0; i < diag.checkpoints.size(); ++i) {
        checkpoints.emplace_back(
            convert_checkpoint(diag.checkpoints[i])
        );
    }
    return py::dict(
        "strong_sizes"_a=diag.strong_sizes,
        "active_sizes"_a=diag.active_sizes,
        "used_strong_rule"_a=diag.used_strong_rule,
        "n_cds"_a=diag.n_cds,
        "n_lambdas_proc"_a=diag.n_lambdas_proc,
        "checkpoints"_a=checkpoints
    );
}

static py::dict transform_data__(
    Eigen::Ref<Eigen::MatrixXd>& X,
    const Eigen::Ref<Eigen::VectorXi>& groups,
    const Eigen::Ref<Eigen::VectorXi>& group_sizes,
    size_t n_threads
)
{
    Eigen::VectorXd diag(X.cols());
    std::vector<Eigen::BDCSVD<Eigen::MatrixXd>> decomps;
    ghostbasil::group_lasso::transform_data(X, groups, group_sizes, n_threads, diag, decomps);
    py::dict d(
        "A_diag"_a=diag
    );
    return d;
}

static py::dict group_basil__(
    Eigen::MatrixXd X,
    const Eigen::Ref<Eigen::VectorXd>& y,
    const Eigen::Ref<Eigen::VectorXi>& groups,
    const Eigen::Ref<Eigen::VectorXi>& group_sizes,
    double alpha,
    const Eigen::Ref<Eigen::VectorXd>& penalty,
    const std::vector<double>& user_lmdas_, // to allow size 0
    size_t max_n_lambdas,
    size_t n_lambdas_iter,
    bool use_strong_rule,
    bool do_early_exit,
    bool verbose_diagnostic,
    size_t delta_strong_size,
    size_t max_strong_size,
    size_t max_n_cds,
    double thr,
    double newton_tol,
    size_t newton_max_iters,
    double min_ratio,
    size_t n_threads
)
{
    using namespace ghostbasil::group_lasso;

    Eigen::Map<const Eigen::VectorXd> user_lmdas(user_lmdas_.data(), user_lmdas_.size());
    std::vector<Eigen::SparseVector<double>> betas_out;
    std::vector<double> lmdas;
    std::vector<double> rsqs_out;
    GroupBasilCheckpoint<double, int, int> checkpoint;
    GroupBasilDiagnostic diagnostic;

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
        update_coefficients(L, v, l1, l2, tol, max_iters, x, iters, buffer1, buffer2);
    };
    
    std::string error;
    try {
        group_basil(
            X, y, groups, group_sizes, alpha, penalty, user_lmdas, 
            max_n_lambdas, n_lambdas_iter, use_strong_rule, do_early_exit, verbose_diagnostic, delta_strong_size, max_strong_size,
            max_n_cds, thr, newton_tol, newton_max_iters, min_ratio, n_threads,
            betas_out, lmdas, rsqs_out, update_coefficients_f, checkpoint, diagnostic
        );
    } catch (const std::exception& e) {
        error = e.what();
    }
    
    Eigen::Map<Eigen::VectorXd> rsqs_map(rsqs_out.data(), rsqs_out.size());
    rsqs_map /= y.squaredNorm();
    
    Eigen::SparseMatrix<double> beta_mat;
    if (betas_out.size()) {
        beta_mat.resize(X.cols(), betas_out.size());
        for (size_t i = 0; i < betas_out.size(); ++i) {
            beta_mat.col(i) = betas_out[i];
        }
    }
    
    return py::dict(
        "betas"_a=beta_mat,
        "lmdas"_a=lmdas,
        "rsqs"_a=rsqs_out,
        "diagnostic"_a=convert_diagnostic(diagnostic),
        "error"_a=error
    );
}