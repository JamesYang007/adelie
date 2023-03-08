#include <Rcpp.h>
#include <RcppEigen.h>
#include <fista.hpp>

//' FISTA solver.
//'
//' @param   L       vector representing a diagonal PSD matrix.
//'                  Must have max(L + s) > 0. 
//'                  L.size() <= buffer1.size().
//' @param   v       any vector.  
//' @param   l1      L2-norm penalty. Must be >= 0.
//' @param   l2      L2 penalty. Must be >= 0.
//' @param   tol         Newton's method tolerance of closeness to 0.
//' @param   max_iters   maximum number of iterations of Newton's method.
//' @export
// [[Rcpp::export]]
Rcpp::List fista_solver(
    const Eigen::Map<Eigen::VectorXd>& L,
    const Eigen::Map<Eigen::VectorXd>& v,
    double l1,
    double l2,
    double tol,
    size_t max_iters
)
{
    Eigen::VectorXd x(L.size());    
    size_t iters = 0;
    glstudy::fista_solver(L, v, l1, l2, tol, max_iters, x, iters);
    return Rcpp::List::create(
        Rcpp::Named("beta")=x,
        Rcpp::Named("iters")=iters
    );
}

//' FISTA solver with adaptive restart.
//'
//' @param   L       vector representing a diagonal PSD matrix.
//'                  Must have max(L + s) > 0. 
//'                  L.size() <= buffer1.size().
//' @param   v       any vector.  
//' @param   l1      L2-norm penalty. Must be >= 0.
//' @param   l2      L2 penalty. Must be >= 0.
//' @param   tol         Newton's method tolerance of closeness to 0.
//' @param   max_iters   maximum number of iterations of Newton's method.
//' @export
// [[Rcpp::export]]
Rcpp::List fista_adares_solver(
    const Eigen::Map<Eigen::VectorXd>& L,
    const Eigen::Map<Eigen::VectorXd>& v,
    double l1,
    double l2,
    double tol,
    size_t max_iters
)
{
    Eigen::VectorXd x(L.size());    
    size_t iters = 0;
    glstudy::fista_adares_solver(L, v, l1, l2, tol, max_iters, x, iters);
    return Rcpp::List::create(
        Rcpp::Named("beta")=x,
        Rcpp::Named("iters")=iters
    );
}