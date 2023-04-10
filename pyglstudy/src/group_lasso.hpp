#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <ghostbasil/optimization/group_lasso_cov.hpp>
#include <ghostbasil/matrix/cov_cache.hpp>

namespace py = pybind11;
using namespace pybind11::literals; // to bring in the `_a` literal

static py::dict group_lasso__(
    const Eigen::Ref<Eigen::MatrixXd>& A,
    const Eigen::Ref<Eigen::VectorXi>& groups, 
    const Eigen::Ref<Eigen::VectorXi>& group_sizes,
    double alpha, 
    const Eigen::Ref<Eigen::VectorXd>& penalty,
    const Eigen::Ref<Eigen::VectorXi>& strong_set, 
    const std::vector<int>& strong_g1,
    const std::vector<int>& strong_g2,
    const Eigen::Ref<Eigen::VectorXi>& strong_begins, 
    const Eigen::Ref<Eigen::VectorXd>& strong_A_diag,
    const Eigen::Ref<Eigen::VectorXd>& lmdas, 
    size_t max_cds,
    double thr,
    double cond_0_thresh,
    double cond_1_thresh,
    double newton_tol,
    size_t newton_max_iters,
    double rsq,
    Eigen::Ref<Eigen::VectorXd>& strong_beta,
    Eigen::Ref<Eigen::VectorXd>& strong_grad,
    std::vector<int> active_set,
    std::vector<int> active_g1,
    std::vector<int> active_g2,
    std::vector<int> active_begins,
    std::vector<int> active_order,
    Eigen::Ref<ghostbasil::util::vec_type<bool>>& is_active
)
{
    using namespace ghostbasil;
    using namespace ghostbasil::group_lasso;
    util::vec_type<util::sp_vec_type<double, Eigen::ColMajor, int>> betas(lmdas.size());
    Eigen::VectorXd rsqs(lmdas.size());

    Eigen::Map<const Eigen::MatrixXd> A_map( 
        A.data(),  
        A.rows(), 
        A.cols()
    );

    cov::GroupLassoParamPack<
        Eigen::Map<const Eigen::MatrixXd>,
        double,
        int,
        bool
    > pack(
        A_map, groups, group_sizes, alpha, penalty, 
        strong_set, strong_g1, strong_g2, strong_begins, strong_A_diag,
        lmdas, max_cds, thr, cond_0_thresh, cond_1_thresh, newton_tol, newton_max_iters, rsq,
        strong_beta, strong_grad, active_set, active_g1,
        active_g2, active_begins, active_order, is_active,
        betas, rsqs, 0, 0
    );
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
        cov::fit(pack, update_coefficients_f);
    } catch(const std::exception& e) {
        error = e.what(); 
    }

    Eigen::SparseMatrix<double> beta_mat;
    if (pack.n_lmdas > 0) {
        beta_mat.resize(A.rows(), pack.n_lmdas);
        for (size_t i = 0; i < pack.n_lmdas; ++i) {
            beta_mat.col(i) = betas[i];
        }
    }

    py::dict d(
        "rsq"_a=rsq,
        "strong_beta"_a=pack.strong_beta,
        "strong_grad"_a=pack.strong_grad,
        "active_set"_a=*pack.active_set,
        "active_g1"_a=*pack.active_g1,
        "active_g2"_a=*pack.active_g2,
        "active_begins"_a=*pack.active_begins,
        "active_order"_a=*pack.active_order,
        "is_active"_a=pack.is_active,
        "lmdas"_a=lmdas.head(pack.n_lmdas),
        "betas"_a=beta_mat,
        "rsqs"_a=pack.rsqs.head(pack.n_lmdas),
        "n_cds"_a=pack.n_cds,
        "n_lmdas"_a=pack.n_lmdas,
        "error"_a=error
    );
    return d;
} 

static py::dict group_lasso_data__(
    const Eigen::Ref<Eigen::MatrixXd>& X,
    const Eigen::Ref<Eigen::VectorXi>& groups, 
    const Eigen::Ref<Eigen::VectorXi>& group_sizes,
    double alpha, 
    const Eigen::Ref<Eigen::VectorXd>& penalty,
    const Eigen::Ref<Eigen::VectorXi>& strong_set, 
    const std::vector<int>& strong_g1,
    const std::vector<int>& strong_g2,
    const Eigen::Ref<Eigen::VectorXi>& strong_begins, 
    const Eigen::Ref<Eigen::VectorXd>& strong_A_diag,
    const Eigen::Ref<Eigen::VectorXd>& lmdas, 
    size_t max_cds,
    double thr,
    double cond_0_thresh,
    double cond_1_thresh,
    double newton_tol,
    size_t newton_max_iters,
    double rsq,
    Eigen::Ref<Eigen::VectorXd>& strong_beta,
    Eigen::Ref<Eigen::VectorXd>& strong_grad,
    std::vector<int> active_set,
    std::vector<int> active_g1,
    std::vector<int> active_g2,
    std::vector<int> active_begins,
    std::vector<int> active_order,
    Eigen::Ref<ghostbasil::util::vec_type<bool>>& is_active
)
{
    using namespace ghostbasil;
    using namespace ghostbasil::group_lasso;
    
    Eigen::Map<const Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    CovCache<Eigen::Map<const Eigen::MatrixXd>, double> A(X_map);
    util::vec_type<util::sp_vec_type<double, Eigen::ColMajor, int>> betas(lmdas.size());
    Eigen::VectorXd rsqs(lmdas.size());

    cov::GroupLassoParamPack<
        std::decay_t<decltype(A)>,
        double,
        int,
        bool
    > pack(
        A, groups, group_sizes, alpha, penalty, 
        strong_set, strong_g1, strong_g2, strong_begins, strong_A_diag,
        lmdas, max_cds, thr, cond_0_thresh, cond_1_thresh, newton_tol, newton_max_iters, rsq,
        strong_beta, strong_grad, active_set, active_g1,
        active_g2, active_begins, active_order, is_active,
        betas, rsqs, 0, 0
    );
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
        cov::fit(pack, update_coefficients_f);
    } catch(const std::exception& e) {
        error = e.what(); 
    }
    
    Eigen::SparseMatrix<double> beta_mat;
    if (pack.n_lmdas > 0) {
        beta_mat.resize(X.cols(), pack.n_lmdas);
        for (size_t i = 0; i < pack.n_lmdas; ++i) {
            beta_mat.col(i) = betas[i];
        }
    }

    py::dict d(
        "rsq"_a=rsq,
        "strong_beta"_a=pack.strong_beta,
        "strong_grad"_a=pack.strong_grad,
        "active_set"_a=*pack.active_set,
        "active_g1"_a=*pack.active_g1,
        "active_g2"_a=*pack.active_g2,
        "active_begins"_a=*pack.active_begins,
        "active_order"_a=*pack.active_order,
        "is_active"_a=pack.is_active,
        "lmdas"_a=lmdas.head(pack.n_lmdas),
        "betas"_a=beta_mat,
        "rsqs"_a=pack.rsqs.head(pack.n_lmdas),
        "n_cds"_a=pack.n_cds,
        "n_lmdas"_a=pack.n_lmdas,
        "error"_a=error
    );
    return d;
} 

static py::dict group_lasso_data_newton__(
    const Eigen::Ref<Eigen::MatrixXd>& X,
    const Eigen::Ref<Eigen::VectorXi>& groups, 
    const Eigen::Ref<Eigen::VectorXi>& group_sizes,
    double alpha, 
    const Eigen::Ref<Eigen::VectorXd>& penalty,
    const Eigen::Ref<Eigen::VectorXi>& strong_set, 
    const std::vector<int>& strong_g1,
    const std::vector<int>& strong_g2,
    const Eigen::Ref<Eigen::VectorXi>& strong_begins, 
    const Eigen::Ref<Eigen::VectorXd>& strong_A_diag,
    const Eigen::Ref<Eigen::VectorXd>& lmdas, 
    size_t max_cds,
    double thr,
    double cond_0_thresh,
    double cond_1_thresh,
    double newton_tol,
    size_t newton_max_iters,
    double rsq,
    Eigen::Ref<Eigen::VectorXd>& strong_beta,
    Eigen::Ref<Eigen::VectorXd>& strong_grad,
    std::vector<int> active_set,
    std::vector<int> active_g1,
    std::vector<int> active_g2,
    std::vector<int> active_begins,
    std::vector<int> active_order,
    Eigen::Ref<ghostbasil::util::vec_type<bool>>& is_active
)
{
    using namespace ghostbasil;
    using namespace ghostbasil::group_lasso;
    
    Eigen::Map<const Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    CovCache<Eigen::Map<const Eigen::MatrixXd>, double> A(X_map);
    util::vec_type<util::sp_vec_type<double, Eigen::ColMajor, int>> betas(lmdas.size());
    Eigen::VectorXd rsqs(lmdas.size());

    cov::GroupLassoParamPack<
        std::decay_t<decltype(A)>,
        double,
        int,
        bool
    > pack(
        A, groups, group_sizes, alpha, penalty, 
        strong_set, strong_g1, strong_g2, strong_begins, strong_A_diag,
        lmdas, max_cds, thr, cond_0_thresh, cond_1_thresh, newton_tol, newton_max_iters, rsq,
        strong_beta, strong_grad, active_set, active_g1,
        active_g2, active_begins, active_order, is_active,
        betas, rsqs, 0, 0
    );
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
        glstudy::newton_solver(L, v, l1, l2, tol, max_iters, x, iters, buffer1, buffer2);
    };

    std::string error;
    try {
        cov::fit(pack, update_coefficients_f);
    } catch(const std::exception& e) {
        error = e.what(); 
    }
    
    Eigen::SparseMatrix<double> beta_mat;
    if (pack.n_lmdas > 0) {
        beta_mat.resize(X.cols(), pack.n_lmdas);
        for (size_t i = 0; i < pack.n_lmdas; ++i) {
            beta_mat.col(i) = betas[i];
        }
    }

    py::dict d(
        "rsq"_a=rsq,
        "strong_beta"_a=pack.strong_beta,
        "strong_grad"_a=pack.strong_grad,
        "active_set"_a=*pack.active_set,
        "active_g1"_a=*pack.active_g1,
        "active_g2"_a=*pack.active_g2,
        "active_begins"_a=*pack.active_begins,
        "active_order"_a=*pack.active_order,
        "is_active"_a=pack.is_active,
        "lmdas"_a=lmdas.head(pack.n_lmdas),
        "betas"_a=beta_mat,
        "rsqs"_a=pack.rsqs.head(pack.n_lmdas),
        "n_cds"_a=pack.n_cds,
        "n_lmdas"_a=pack.n_lmdas,
        "error"_a=error
    );
    return d;
} 