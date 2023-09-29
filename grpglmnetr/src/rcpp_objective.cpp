#include <Rcpp.h>
#include <RcppEigen.h>
#include <objective.hpp>

//' Computes the group-lasso objective.
//' 
//' @param   A       any square (p, p) matrix. 
//' @param   r       any vector (p,).
//' @param   groups  vector defining group beginning indices.
//' @param   group_sizes vector defining group sizes.
//' @param   alpha       elastic net proportion.
//' @param   penalty penalty factor for each group.
//' @param   lmda    group-lasso regularization.
//' @param   beta    coefficient vector.
//' @export
// [[Rcpp::export]]
double objective(
    const Eigen::Map<Eigen::MatrixXd>& A,
    const Eigen::Map<Eigen::VectorXd>& r,
    const Eigen::Map<Eigen::VectorXi>& groups,
    const Eigen::Map<Eigen::VectorXi>& group_sizes,
    double alpha,
    const Eigen::Map<Eigen::VectorXd>& penalty,
    double lmda,
    const Eigen::Map<Eigen::VectorXd>& beta
)
{
    return grpglmnet_core::objective(A, r, groups, group_sizes, alpha, penalty, lmda, beta);
}