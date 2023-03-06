#include <Rcpp.h>
#include <RcppEigen.h>
#include <newton.hpp>

//' Solves the solution for the equation (w.r.t. \f$x\f$):
//'
//' @param   L       vector representing a diagonal PSD matrix.
//'                  Must have max(L + s) > 0. 
//'                  L.size() <= buffer1.size().
//' @param   v       any vector.  
//' @param   l1      L2-norm penalty. Must be >= 0.
//' @param   l2      L2 penalty. Must be >= 0.
//' @param   tol         Newton's method tolerance of closeness to 0.
//' @param   max_iters   maximum number of iterations of Newton's method.
//' @param   x           solution vector.
//' @param   iters       number of Newton's method iterations taken.
//' @param   buffer1     any vector with L.size() <= buffer1.size().
//' @param   buffer2     any vector with L.size() <= buffer2.size().
//' @export
// [[Rcpp::export]]
Rcpp::List newton_solver(
    const Eigen::Map<Eigen::VectorXd>& L,
    const Eigen::Map<Eigen::VectorXd>& v,
    double l1,
    double l2,
    double tol,
    size_t max_iters
)
{
    Eigen::VectorXd x(L.size());
    Eigen::VectorXd buffer1(L.size());
    Eigen::VectorXd buffer2(L.size());
    size_t iters = 0;
    glstudy::newton_solver(L, v, l1, l2, tol, max_iters, x, iters, buffer1, buffer2);
    return Rcpp::List::create(
        Rcpp::Named("beta")=x,
        Rcpp::Named("iters")=iters
    );
}