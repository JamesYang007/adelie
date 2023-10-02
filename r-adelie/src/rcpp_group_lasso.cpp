#include <Rcpp.h>
#include <RcppEigen.h>
#include <adelie_core/optimization/group_basil_cov.hpp>
#include <adelie_core/optimization/group_elnet_cov.hpp>
#include <adelie_core/optimization/group_basil_naive.hpp>
#include <adelie_core/optimization/group_elnet_naive.hpp>

namespace gl_naive = adelie_core::naive;
namespace gl_cov = adelie_core::cov;

//' @export
// [[Rcpp::export]]
Rcpp::List group_basil_cov__(
    const Eigen::Map<Eigen::MatrixXd>& X,
    const Eigen::Map<Eigen::VectorXd>& y,
    const Eigen::Map<Eigen::VectorXi>& groups,
    const Eigen::Map<Eigen::VectorXi>& group_sizes,
    double alpha,
    const Eigen::Map<Eigen::VectorXd>& penalty,
    const Eigen::Map<Eigen::VectorXd>& user_lmdas_, // to allow size 0
    size_t max_n_lambdas,
    size_t n_lambdas_iter,
    bool use_strong_rule,
    bool do_early_exit,
    bool verbose_diagnostic,
    size_t delta_strong_size,
    size_t max_strong_size,
    size_t max_n_cds,
    double tol,
    double rsq_slope_tol,
    double rsq_curv_tol,
    double newton_tol,
    size_t newton_max_iters,
    double min_ratio,
    size_t n_threads
)
{
    Eigen::Map<const Eigen::VectorXd> user_lmdas(user_lmdas_.data(), user_lmdas_.size());
    std::vector<Eigen::SparseVector<double>> betas_out;
    std::vector<double> lmdas;
    std::vector<double> rsqs_out;
    gl_cov::GroupBasilCheckpoint<double, int, int> checkpoint;
    gl_cov::GroupBasilDiagnostic diagnostic;

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
        gl_cov::group_basil(
            X, y, groups, group_sizes, alpha, penalty, user_lmdas, 
            max_n_lambdas, n_lambdas_iter, use_strong_rule, do_early_exit, verbose_diagnostic, delta_strong_size, max_strong_size,
            max_n_cds, tol, rsq_slope_tol, rsq_curv_tol, newton_tol, newton_max_iters, min_ratio, n_threads,
            betas_out, lmdas, rsqs_out, update_coefficients_f, checkpoint, diagnostic
        );
    } catch (const std::exception& e) {
        error = e.what();
    }
    
    Eigen::Map<Eigen::VectorXd> rsqs_map(rsqs_out.data(), rsqs_out.size());
    rsqs_map /= y.squaredNorm();
    
    Eigen::SparseMatrix<double> beta_mat;
    beta_mat.setZero();
    if (betas_out.size()) {
        beta_mat.resize(X.cols(), betas_out.size());
        for (size_t i = 0; i < betas_out.size(); ++i) {
            beta_mat.col(i) = betas_out[i];
        }
    }
    
    return Rcpp::List::create(
        Rcpp::Named("betas")=beta_mat,
        Rcpp::Named("lmdas")=lmdas,
        Rcpp::Named("rsqs")=rsqs_out,
        Rcpp::Named("error")=error
    );
}

//' @export
// [[Rcpp::export]]
Rcpp::List group_basil_naive__(
    const Eigen::Map<Eigen::MatrixXd>& X,
    const Eigen::Map<Eigen::VectorXd>& y,
    const Eigen::Map<Eigen::VectorXi>& groups,
    const Eigen::Map<Eigen::VectorXi>& group_sizes,
    double alpha,
    const Eigen::Map<Eigen::VectorXd>& penalty,
    const Eigen::Map<Eigen::VectorXd>& user_lmdas_, // to allow size 0
    size_t max_n_lambdas,
    size_t n_lambdas_iter,
    bool use_strong_rule,
    bool do_early_exit,
    bool verbose_diagnostic,
    size_t delta_strong_size,
    size_t max_strong_size,
    size_t max_n_cds,
    double tol,
    double rsq_slope_tol,
    double rsq_curv_tol,
    double newton_tol,
    size_t newton_max_iters,
    double min_ratio,
    size_t n_threads
)
{
    Eigen::Map<const Eigen::VectorXd> user_lmdas(user_lmdas_.data(), user_lmdas_.size());
    std::vector<Eigen::SparseVector<double>> betas_out;
    std::vector<double> lmdas;
    std::vector<double> rsqs_out;
    gl_naive::GroupBasilCheckpoint<double, int, int> checkpoint;
    gl_naive::GroupBasilDiagnostic diagnostic;

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
        gl_naive::group_basil(
            X, y, groups, group_sizes, alpha, penalty, user_lmdas, 
            max_n_lambdas, n_lambdas_iter, use_strong_rule, do_early_exit, verbose_diagnostic, delta_strong_size, max_strong_size,
            max_n_cds, tol, rsq_slope_tol, rsq_curv_tol, newton_tol, newton_max_iters, min_ratio, n_threads,
            betas_out, lmdas, rsqs_out, update_coefficients_f, checkpoint, diagnostic
        );
    } catch (const std::exception& e) {
        error = e.what();
    }
    
    Eigen::Map<Eigen::VectorXd> rsqs_map(rsqs_out.data(), rsqs_out.size());
    rsqs_map /= y.squaredNorm();
    
    Eigen::SparseMatrix<double> beta_mat;
    beta_mat.setZero();
    if (betas_out.size()) {
        beta_mat.resize(X.cols(), betas_out.size());
        for (size_t i = 0; i < betas_out.size(); ++i) {
            beta_mat.col(i) = betas_out[i];
        }
    }
    
    return Rcpp::List::create(
        Rcpp::Named("betas")=beta_mat,
        Rcpp::Named("lmdas")=lmdas,
        Rcpp::Named("rsqs")=rsqs_out,
        Rcpp::Named("error")=error
    );
}