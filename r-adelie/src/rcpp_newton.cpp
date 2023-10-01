#include <Rcpp.h>
#include <RcppEigen.h>
#include <newton.hpp>

//' Newton solver
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
    adelie_core::newton_solver(L, v, l1, l2, tol, max_iters, x, iters, buffer1, buffer2);
    return Rcpp::List::create(
        Rcpp::Named("beta")=x,
        Rcpp::Named("iters")=iters
    );
}

//' Newton-ABS solver
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
Rcpp::List newton_abs_solver(
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
    adelie_core::newton_abs_solver(L, v, l1, l2, tol, max_iters, x, iters, buffer1, buffer2);
    return Rcpp::List::create(
        Rcpp::Named("beta")=x,
        Rcpp::Named("iters")=iters
    );
}

//' Newton solver with more information.
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
Rcpp::List newton_abs_debug_solver(
    const Eigen::Map<Eigen::VectorXd>& L,
    const Eigen::Map<Eigen::VectorXd>& v,
    double l1,
    double l2,
    double tol,
    size_t max_iters,
    bool smart_init=true
)
{
    double h_min = 0;
    double h_max = std::numeric_limits<double>::infinity();
    std::vector<double> iters;
    iters.reserve(max_iters);
    std::vector<double> smart_iters;
    smart_iters.reserve(max_iters);
    Eigen::VectorXd x(L.size());
    Eigen::VectorXd buffer1(L.size());
    Eigen::VectorXd buffer2(L.size());
    adelie_core::newton_abs_debug_solver(L, v, l1, l2, tol, max_iters, smart_init, h_min, h_max, x, iters, smart_iters, buffer1, buffer2);
    return Rcpp::List::create(
        Rcpp::Named("beta")=x,
        Rcpp::Named("h_min")=h_min,
        Rcpp::Named("h_max")=h_max,
        Rcpp::Named("iters")=iters,
        Rcpp::Named("smart_iters")=smart_iters
    );
}