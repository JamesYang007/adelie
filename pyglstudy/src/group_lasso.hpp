#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <ghostbasil/optimization/group_lasso.hpp>

namespace py = pybind11;
using namespace pybind11::literals; // to bring in the `_a` literal

// NOTE: all active vectors must be prepended with a dummy value.
// This is because pybind11 is so stupid. It doesn't know how to pass empty arrays smh.
// An array of size of 1 for active vectors is considered to be the empty vector.
static py::dict group_lasso(
    const Eigen::Ref<Eigen::MatrixXd>& A,
    const Eigen::Ref<Eigen::VectorXi>& groups, 
    const Eigen::Ref<Eigen::VectorXi>& group_sizes,
    double alpha, 
    const Eigen::Ref<Eigen::VectorXd>& penalty,
    const Eigen::Ref<Eigen::VectorXi>& strong_set, 
    const Eigen::Ref<Eigen::VectorXi>& strong_g1,
    const Eigen::Ref<Eigen::VectorXi>& strong_g2,
    const Eigen::Ref<Eigen::VectorXi>& strong_begins, 
    const Eigen::Ref<Eigen::VectorXd>& strong_A_diag,
    const Eigen::Ref<Eigen::VectorXd>& lmdas, 
    size_t max_cds,
    double thr,
    double newton_tol,
    size_t newton_max_iters,
    double rsq,
    Eigen::Ref<Eigen::VectorXd>& strong_beta_,
    Eigen::Ref<Eigen::VectorXd>& strong_grad_,
    const Eigen::Ref<Eigen::VectorXi>& active_set_,
    const Eigen::Ref<Eigen::VectorXi>& active_g1_,
    const Eigen::Ref<Eigen::VectorXi>& active_g2_,
    const Eigen::Ref<Eigen::VectorXi>& active_begins_,
    const Eigen::Ref<Eigen::VectorXi>& active_order_,
    Eigen::Ref<ghostbasil::util::vec_type<bool>>& is_active_
)
{
    using namespace ghostbasil;
    using namespace ghostbasil::group_lasso;
    std::vector<int> active_set(active_set_.data()+1, active_set_.data()+active_set_.size());
    std::vector<int> active_g1(active_g1_.data()+1, active_g1_.data()+active_g1_.size());
    std::vector<int> active_g2(active_g2_.data()+1, active_g2_.data()+active_g2_.size());
    std::vector<int> active_begins(active_begins_.data()+1, active_begins_.data()+active_begins_.size());
    std::vector<int> active_order(active_order_.data()+1, active_order_.data()+active_order_.size());
    std::vector<util::sp_vec_type<double, Eigen::ColMajor, int>> betas(lmdas.size());
    std::vector<double> rsqs(lmdas.size());
    Eigen::Map<Eigen::VectorXd> strong_beta(strong_beta_.data(), strong_beta_.size());
    Eigen::Map<Eigen::VectorXd> strong_grad(strong_grad_.data(), strong_grad_.size());
    Eigen::Map<util::vec_type<bool>> is_active(is_active_.data(), is_active_.size());

    Eigen::Map<const Eigen::MatrixXd> A_map( 
        A.data(),  
        A.rows(), 
        A.cols()
    );

    GroupLassoParamPack<
        Eigen::Map<const Eigen::MatrixXd>,
        double,
        int,
        bool
    > pack(
        A_map, groups, group_sizes, alpha, penalty, 
        strong_set, strong_g1, strong_g2, strong_begins, strong_A_diag,
        lmdas, max_cds, thr, newton_tol, newton_max_iters, rsq,
        strong_beta, strong_grad, active_set, active_g1,
        active_g2, active_begins, active_order, is_active,
        betas, rsqs, 0, 0
    );
    fit(pack);

    Eigen::SparseMatrix<double> beta_mat;
    if (pack.n_lmdas > 0) {
        beta_mat.resize(A.rows(), pack.n_lmdas);
        for (size_t i = 0; i < pack.n_lmdas; ++i) {
            beta_mat.col(i) = betas[i];
        }
    }

    py::dict d(
        "strong_beta"_a=pack.strong_beta,
        "strong_grad"_a=pack.strong_grad,
        "active_set"_a=*pack.active_set,
        "active_g1"_a=*pack.active_g1,
        "active_g2"_a=*pack.active_g2,
        "active_begins"_a=*pack.active_begins,
        "active_order"_a=*pack.active_order,
        "is_active"_a=pack.is_active,
        "betas"_a=beta_mat,
        "rsqs"_a=pack.rsqs,
        "n_cds"_a=pack.n_cds,
        "n_lmdas"_a=pack.n_lmdas
    );
    return d;
} 
