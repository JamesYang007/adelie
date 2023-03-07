// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// fista_solver
Rcpp::List fista_solver(const Eigen::Map<Eigen::VectorXd>& L, const Eigen::Map<Eigen::VectorXd>& v, double l1, double l2, double tol, size_t max_iters);
RcppExport SEXP _glstudy_fista_solver(SEXP LSEXP, SEXP vSEXP, SEXP l1SEXP, SEXP l2SEXP, SEXP tolSEXP, SEXP max_itersSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd>& >::type L(LSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd>& >::type v(vSEXP);
    Rcpp::traits::input_parameter< double >::type l1(l1SEXP);
    Rcpp::traits::input_parameter< double >::type l2(l2SEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< size_t >::type max_iters(max_itersSEXP);
    rcpp_result_gen = Rcpp::wrap(fista_solver(L, v, l1, l2, tol, max_iters));
    return rcpp_result_gen;
END_RCPP
}
// newton_solver
Rcpp::List newton_solver(const Eigen::Map<Eigen::VectorXd>& L, const Eigen::Map<Eigen::VectorXd>& v, double l1, double l2, double tol, size_t max_iters);
RcppExport SEXP _glstudy_newton_solver(SEXP LSEXP, SEXP vSEXP, SEXP l1SEXP, SEXP l2SEXP, SEXP tolSEXP, SEXP max_itersSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd>& >::type L(LSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd>& >::type v(vSEXP);
    Rcpp::traits::input_parameter< double >::type l1(l1SEXP);
    Rcpp::traits::input_parameter< double >::type l2(l2SEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< size_t >::type max_iters(max_itersSEXP);
    rcpp_result_gen = Rcpp::wrap(newton_solver(L, v, l1, l2, tol, max_iters));
    return rcpp_result_gen;
END_RCPP
}
// objective
double objective(const Eigen::Map<Eigen::MatrixXd>& A, const Eigen::Map<Eigen::VectorXd>& r, const Eigen::Map<Eigen::VectorXi>& groups, const Eigen::Map<Eigen::VectorXi>& group_sizes, double alpha, const Eigen::Map<Eigen::VectorXd>& penalty, double lmda, const Eigen::Map<Eigen::VectorXd>& beta);
RcppExport SEXP _glstudy_objective(SEXP ASEXP, SEXP rSEXP, SEXP groupsSEXP, SEXP group_sizesSEXP, SEXP alphaSEXP, SEXP penaltySEXP, SEXP lmdaSEXP, SEXP betaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type A(ASEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd>& >::type r(rSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXi>& >::type groups(groupsSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXi>& >::type group_sizes(group_sizesSEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd>& >::type penalty(penaltySEXP);
    Rcpp::traits::input_parameter< double >::type lmda(lmdaSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd>& >::type beta(betaSEXP);
    rcpp_result_gen = Rcpp::wrap(objective(A, r, groups, group_sizes, alpha, penalty, lmda, beta));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_glstudy_fista_solver", (DL_FUNC) &_glstudy_fista_solver, 6},
    {"_glstudy_newton_solver", (DL_FUNC) &_glstudy_newton_solver, 6},
    {"_glstudy_objective", (DL_FUNC) &_glstudy_objective, 8},
    {NULL, NULL, 0}
};

RcppExport void R_init_glstudy(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
