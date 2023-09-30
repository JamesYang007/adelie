#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <grpglmnet_core/optimization/block_update.hpp>
//#include <grpglmnet_core/optimization/group_elnet_cov.hpp>
//#include <grpglmnet_core/matrix/cov_cache.hpp>
#include <grpglmnet_core/optimization/group_elnet_naive.hpp>

namespace py = pybind11;
namespace gg = grpglmnet_core;
using namespace pybind11::literals; // to bring in the `_a` literal

// =================================================================
// Block Update Methods
// =================================================================

static py::dict ista_solver(
    Eigen::Ref<gg::util::rowvec_type<double>> L,
    Eigen::Ref<gg::util::rowvec_type<double>> v,
    double l1,
    double l2,
    double tol,
    size_t max_iters
)
{
    gg::util::rowvec_type<double> x(L.size());
    size_t iters = 0;
    gg::ista_solver(L, v, l1, l2, tol, max_iters, x, iters);
    py::dict d("beta"_a=x, "iters"_a=iters);
    return d;
}

static py::dict fista_solver(
    Eigen::Ref<gg::util::rowvec_type<double>> L,
    Eigen::Ref<gg::util::rowvec_type<double>> v,
    double l1,
    double l2,
    double tol,
    size_t max_iters
)
{
    gg::util::rowvec_type<double> x(L.size());
    size_t iters = 0;
    gg::fista_solver(L, v, l1, l2, tol, max_iters, x, iters);
    py::dict d("beta"_a=x, "iters"_a=iters);
    return d;
}

static py::dict fista_adares_solver(
    Eigen::Ref<gg::util::rowvec_type<double>> L,
    Eigen::Ref<gg::util::rowvec_type<double>> v,
    double l1,
    double l2,
    double tol,
    size_t max_iters
)
{
    gg::util::rowvec_type<double> x(L.size());
    size_t iters = 0;
    gg::fista_adares_solver(L, v, l1, l2, tol, max_iters, x, iters);
    py::dict d("beta"_a=x, "iters"_a=iters);
    return d;
}

static double bcd_root_lower_bound(
    const Eigen::Ref<gg::util::rowvec_type<double>>& quad,
    const Eigen::Ref<gg::util::rowvec_type<double>>& linear,
    double l1
) 
{
    return gg::bcd_root_lower_bound(quad, linear, l1); 
}

static double bcd_root_upper_bound(
    const Eigen::Ref<gg::util::rowvec_type<double>>& quad,
    const Eigen::Ref<gg::util::rowvec_type<double>>& linear,
    double zero_tol=1e-10
)
{
    const auto out = gg::bcd_root_upper_bound(quad, linear, zero_tol); 
    return std::get<0>(out);
}

static double bcd_root_function(
    double h,
    const Eigen::Ref<gg::util::rowvec_type<double>>& D,
    const Eigen::Ref<gg::util::rowvec_type<double>>& v,
    double l1
)
{
    return gg::bcd_root_function(h, D, v, l1);
}

static py::dict newton_solver(
    Eigen::Ref<gg::util::rowvec_type<double>> L,
    Eigen::Ref<gg::util::rowvec_type<double>> v,
    double l1,
    double l2,
    double tol,
    size_t max_iters
)
{
    gg::util::rowvec_type<double> x(L.size());
    gg::util::rowvec_type<double> buffer1(L.size());
    gg::util::rowvec_type<double> buffer2(L.size());
    size_t iters = 0;
    gg::newton_solver(L, v, l1, l2, tol, max_iters, x, iters, buffer1, buffer2);

    py::dict d("beta"_a=x, "iters"_a=iters);
    return d;
}

static py::dict newton_brent_solver(
    Eigen::Ref<gg::util::rowvec_type<double>> L,
    Eigen::Ref<gg::util::rowvec_type<double>> v,
    double l1,
    double l2,
    double tol,
    size_t max_iters
)
{
    gg::util::rowvec_type<double> x(L.size());
    gg::util::rowvec_type<double> buffer1(L.size());
    gg::util::rowvec_type<double> buffer2(L.size());
    size_t iters = 0;
    gg::newton_brent_solver(L, v, l1, l2, tol, tol, max_iters, x, iters, buffer1, buffer2);

    py::dict d("beta"_a=x, "iters"_a=iters);
    return d;
}

static py::dict newton_abs_solver(
    Eigen::Ref<gg::util::rowvec_type<double>> L,
    Eigen::Ref<gg::util::rowvec_type<double>> v,
    double l1,
    double l2,
    double tol,
    size_t max_iters
)
{
    gg::util::rowvec_type<double> x(L.size());
    gg::util::rowvec_type<double> buffer1(L.size());
    gg::util::rowvec_type<double> buffer2(L.size());
    size_t iters = 0;
    gg::newton_abs_solver(L, v, l1, l2, tol, max_iters, x, iters, buffer1, buffer2);

    py::dict d("beta"_a=x, "iters"_a=iters);
    return d;
}

static py::dict newton_abs_debug_solver(
    Eigen::Ref<gg::util::rowvec_type<double>> L,
    Eigen::Ref<gg::util::rowvec_type<double>> v,
    double l1,
    double l2,
    double tol,
    size_t max_iters,
    bool smart_init
)
{
    double h_min, h_max;
    gg::util::rowvec_type<double> x(L.size());
    gg::util::rowvec_type<double> buffer1(L.size());
    gg::util::rowvec_type<double> buffer2(L.size());
    std::vector<double> iters;
    iters.reserve(L.size());
    std::vector<double> smart_iters;
    smart_iters.reserve(L.size());
    gg::newton_abs_debug_solver(
        L, v, l1, l2, tol, max_iters, smart_init, 
        h_min, h_max, x, iters, smart_iters, buffer1, buffer2
    );
    
    py::dict d(
        "beta"_a=x,
        "h_min"_a=h_min,
        "h_max"_a=h_max,
        "iters"_a=iters,
        "smart_iters"_a=smart_iters
    );
    return d;
}

static py::dict brent_solver(
    Eigen::Ref<gg::util::rowvec_type<double>> L,
    Eigen::Ref<gg::util::rowvec_type<double>> v,
    double l1,
    double l2,
    double tol,
    size_t max_iters
)
{
    gg::util::rowvec_type<double> x(v.size());
    size_t iters = 0;
    gg::brent_solver(L, v, l1, l2, tol, max_iters, x, iters);
    py::dict d("beta"_a=x, "iters"_a=iters);
    return d;
}

// =================================================================
// Group Elastic Net
// =================================================================

template <class XType>
static void register_group_elnet_state(
    py::module_& m,
    const char* name
)
{
    using namespace grpglmnet_core;
    using gls_t = naive::GroupElnetState<XType>;
    using value_t = typename gls_t::value_t;
    using vec_index_t = typename gls_t::vec_index_t;
    using vec_value_t = typename gls_t::vec_value_t;
    using vec_bool_t = typename gls_t::vec_bool_t;
    using dyn_vec_index_t = typename gls_t::dyn_vec_index_t;
    using dyn_vec_value_t = typename gls_t::dyn_vec_value_t;
    using dyn_vec_vec_value_t = typename gls_t::dyn_vec_vec_value_t;
    using dyn_vec_sp_vec_t = typename gls_t::dyn_vec_sp_vec_t;
    py::class_<gls_t>(m, name)
        .def(py::init<
            const XType&,
            const Eigen::Ref<const vec_index_t>&, 
            const Eigen::Ref<const vec_index_t>&,
            value_t,
            const Eigen::Ref<const vec_value_t>&, 
            const Eigen::Ref<const vec_index_t>&, 
            const Eigen::Ref<const vec_index_t>&, 
            const Eigen::Ref<const vec_index_t>&, 
            const Eigen::Ref<const vec_index_t>&, 
            const Eigen::Ref<const vec_value_t>&, 
            const Eigen::Ref<const vec_value_t>&, 
            size_t,
            value_t,
            value_t,
            value_t,
            value_t,
            size_t,
            value_t,
            Eigen::Ref<vec_value_t>,
            Eigen::Ref<vec_value_t>,
            Eigen::Ref<vec_value_t>,
            dyn_vec_index_t, 
            dyn_vec_index_t, 
            dyn_vec_index_t, 
            dyn_vec_index_t, 
            dyn_vec_index_t, 
            Eigen::Ref<vec_bool_t>,
            dyn_vec_sp_vec_t,
            dyn_vec_value_t,
            dyn_vec_vec_value_t
        >(),
            py::arg("X"),
            py::arg("groups"),
            py::arg("group_sizes"),
            py::arg("alpha"),
            py::arg("penalty"),
            py::arg("strong_set"),
            py::arg("strong_g1"),
            py::arg("strong_g2"),
            py::arg("strong_begins"),
            py::arg("strong_A_diag"),
            py::arg("lmdas"),
            py::arg("max_cds"),
            py::arg("thr"),
            py::arg("cond_0_thresh"),
            py::arg("cond_1_thresh"),
            py::arg("newton_tol"),
            py::arg("newton_max_iters"),
            py::arg("rsq"),
            py::arg("resid"),
            py::arg("strong_beta"),
            py::arg("strong_grad"),
            py::arg("active_set"),
            py::arg("active_g1"),
            py::arg("active_g2"),
            py::arg("active_begins"),
            py::arg("active_order"),
            py::arg("is_active"),
            py::arg("betas"),
            py::arg("rsqs"),
            py::arg("resids")
        )
        .def_readonly("groups", &gls_t::groups)
        .def_readonly("group_sizes", &gls_t::group_sizes)
        .def_readonly("alpha", &gls_t::alpha)
        .def_readonly("penalty", &gls_t::penalty)
        .def_readonly("strong_set", &gls_t::strong_set)
        .def_readonly("strong_g1", &gls_t::strong_g1)
        .def_readonly("strong_g2", &gls_t::strong_g2)
        .def_readonly("strong_begins", &gls_t::strong_begins)
        .def_readonly("strong_A_diag", &gls_t::strong_A_diag)
        .def_readonly("lmdas", &gls_t::lmdas)
        .def_readonly("max_cds", &gls_t::max_cds)
        .def_readonly("thr", &gls_t::thr)
        .def_readonly("cond_0_thresh", &gls_t::cond_0_thresh)
        .def_readonly("cond_1_thresh", &gls_t::cond_1_thresh)
        .def_readonly("newton_tol", &gls_t::newton_tol)
        .def_readonly("newton_max_iters", &gls_t::newton_max_iters)
        .def_readonly("rsq", &gls_t::rsq)
        .def_readonly("resid", &gls_t::resid)
        .def_readonly("strong_beta", &gls_t::strong_beta)
        .def_readonly("strong_grad", &gls_t::strong_grad)
        .def_readonly("active_set", &gls_t::active_set)
        .def_readonly("active_g1", &gls_t::active_g1)
        .def_readonly("active_g2", &gls_t::active_g2)
        .def_readonly("active_begins", &gls_t::active_begins)
        .def_readonly("active_order", &gls_t::active_order)
        .def_readonly("is_active", &gls_t::is_active)
        .def_property_readonly("betas", [](const gls_t& obj){
            const auto& betas = obj.betas; 
            const auto p = (betas.size() == 0) ? 0 : betas.back().size();
            util::rowmat_type<value_t> out(betas.size(), p);
            for (size_t i = 0; i < betas.size(); ++i) {
                out.row(i) = betas[i];
            }
            return out;
        })
        .def_readonly("rsqs", &gls_t::rsqs)
        .def_readonly("resids", &gls_t::resids)
        .def_readonly("n_cds", &gls_t::n_cds)
        ;
}

static double group_elnet_objective(
    const Eigen::Ref<gg::util::rowvec_type<double>>& beta,
    const Eigen::Ref<gg::util::rowmat_type<double>>& X,
    const Eigen::Ref<gg::util::rowvec_type<double>>& y,
    const Eigen::Ref<gg::util::rowvec_type<int>>& groups,
    const Eigen::Ref<gg::util::rowvec_type<int>>& group_sizes,
    double lmda,
    double alpha,
    const Eigen::Ref<gg::util::rowvec_type<double>>& penalty
)
{
    return grpglmnet_core::group_elnet_objective(
        beta, X, y, groups, group_sizes, lmda, alpha, penalty
    );
}

static py::dict group_elnet_naive_dense(
    gg::naive::GroupElnetState<
        Eigen::Ref<const gg::util::rowmat_type<double>>
    > pack
)
{
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
        gg::update_coefficients(L, v, l1, l2, tol, max_iters, x, iters, buffer1, buffer2);
    };

    std::string error;
    try {
        gg::naive::fit(pack, update_coefficients_f);
    } catch(const std::exception& e) {
        error = e.what(); 
    }

    return py::dict("state"_a=pack, "error"_a=error);
} 

//static py::dict group_elnet__(
//    const Eigen::Ref<Eigen::MatrixXd>& A,
//    const Eigen::Ref<Eigen::VectorXi>& groups, 
//    const Eigen::Ref<Eigen::VectorXi>& group_sizes,
//    double alpha, 
//    const Eigen::Ref<Eigen::VectorXd>& penalty,
//    const Eigen::Ref<Eigen::VectorXi>& strong_set, 
//    const std::vector<int>& strong_g1,
//    const std::vector<int>& strong_g2,
//    const Eigen::Ref<Eigen::VectorXi>& strong_begins, 
//    const Eigen::Ref<Eigen::VectorXd>& strong_A_diag,
//    const Eigen::Ref<Eigen::VectorXd>& lmdas, 
//    size_t max_cds,
//    double thr,
//    double cond_0_thresh,
//    double cond_1_thresh,
//    double newton_tol,
//    size_t newton_max_iters,
//    double rsq,
//    Eigen::Ref<Eigen::VectorXd>& strong_beta,
//    Eigen::Ref<Eigen::VectorXd>& strong_grad,
//    std::vector<int> active_set,
//    std::vector<int> active_g1,
//    std::vector<int> active_g2,
//    std::vector<int> active_begins,
//    std::vector<int> active_order,
//    Eigen::Ref<grpglmnet_core::util::vec_type<bool>>& is_active
//)
//{
//    using namespace grpglmnet_core;
//    util::vec_type<util::sp_vec_type<double, Eigen::ColMajor, int>> betas(lmdas.size());
//    Eigen::VectorXd rsqs(lmdas.size());
//
//    Eigen::Map<const Eigen::MatrixXd> A_map( 
//        A.data(),  
//        A.rows(), 
//        A.cols()
//    );
//
//    cov::GroupElnetParamPack<
//        Eigen::Map<const Eigen::MatrixXd>,
//        double,
//        int,
//        bool
//    > pack(
//        A_map, groups, group_sizes, alpha, penalty, 
//        strong_set, strong_g1, strong_g2, strong_begins, strong_A_diag,
//        lmdas, max_cds, thr, cond_0_thresh, cond_1_thresh, newton_tol, newton_max_iters, rsq,
//        strong_beta, strong_grad, active_set, active_g1,
//        active_g2, active_begins, active_order, is_active,
//        betas, rsqs, 0, 0
//    );
//    const auto update_coefficients_f = [](
//        const auto& L,
//        const auto& v,
//        auto l1,
//        auto l2,
//        auto tol,
//        size_t max_iters,
//        auto& x,
//        auto& iters,
//        auto& buffer1,
//        auto& buffer2
//    ){
//        update_coefficients(L, v, l1, l2, tol, max_iters, x, iters, buffer1, buffer2);
//    };
//
//    std::string error;
//    try {
//        cov::fit(pack, update_coefficients_f);
//    } catch(const std::exception& e) {
//        error = e.what(); 
//    }
//
//    Eigen::SparseMatrix<double> beta_mat;
//    if (pack.n_lmdas > 0) {
//        beta_mat.resize(A.rows(), pack.n_lmdas);
//        for (size_t i = 0; i < pack.n_lmdas; ++i) {
//            beta_mat.col(i) = betas[i];
//        }
//    }
//
//    py::dict d(
//        "rsq"_a=rsq,
//        "strong_beta"_a=pack.strong_beta,
//        "strong_grad"_a=pack.strong_grad,
//        "active_set"_a=*pack.active_set,
//        "active_g1"_a=*pack.active_g1,
//        "active_g2"_a=*pack.active_g2,
//        "active_begins"_a=*pack.active_begins,
//        "active_order"_a=*pack.active_order,
//        "is_active"_a=pack.is_active,
//        "lmdas"_a=lmdas.head(pack.n_lmdas),
//        "betas"_a=beta_mat,
//        "rsqs"_a=pack.rsqs.head(pack.n_lmdas),
//        "n_cds"_a=pack.n_cds,
//        "n_lmdas"_a=pack.n_lmdas,
//        "error"_a=error
//    );
//    return d;
//} 
//
//static py::dict group_elnet_data__(
//    const Eigen::Ref<Eigen::MatrixXd>& X,
//    const Eigen::Ref<Eigen::VectorXi>& groups, 
//    const Eigen::Ref<Eigen::VectorXi>& group_sizes,
//    double alpha, 
//    const Eigen::Ref<Eigen::VectorXd>& penalty,
//    const Eigen::Ref<Eigen::VectorXi>& strong_set, 
//    const std::vector<int>& strong_g1,
//    const std::vector<int>& strong_g2,
//    const Eigen::Ref<Eigen::VectorXi>& strong_begins, 
//    const Eigen::Ref<Eigen::VectorXd>& strong_A_diag,
//    const Eigen::Ref<Eigen::VectorXd>& lmdas, 
//    size_t max_cds,
//    double thr,
//    double cond_0_thresh,
//    double cond_1_thresh,
//    double newton_tol,
//    size_t newton_max_iters,
//    double rsq,
//    Eigen::Ref<Eigen::VectorXd>& strong_beta,
//    Eigen::Ref<Eigen::VectorXd>& strong_grad,
//    std::vector<int> active_set,
//    std::vector<int> active_g1,
//    std::vector<int> active_g2,
//    std::vector<int> active_begins,
//    std::vector<int> active_order,
//    Eigen::Ref<grpglmnet_core::util::vec_type<bool>>& is_active
//)
//{
//    using namespace grpglmnet_core;
//    
//    Eigen::Map<const Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
//    CovCache<Eigen::Map<const Eigen::MatrixXd>, double> A(X_map);
//    util::vec_type<util::sp_vec_type<double, Eigen::ColMajor, int>> betas(lmdas.size());
//    Eigen::VectorXd rsqs(lmdas.size());
//
//    cov::GroupElnetParamPack<
//        std::decay_t<decltype(A)>,
//        double,
//        int,
//        bool
//    > pack(
//        A, groups, group_sizes, alpha, penalty, 
//        strong_set, strong_g1, strong_g2, strong_begins, strong_A_diag,
//        lmdas, max_cds, thr, cond_0_thresh, cond_1_thresh, newton_tol, newton_max_iters, rsq,
//        strong_beta, strong_grad, active_set, active_g1,
//        active_g2, active_begins, active_order, is_active,
//        betas, rsqs, 0, 0
//    );
//    const auto update_coefficients_f = [](
//        const auto& L,
//        const auto& v,
//        auto l1,
//        auto l2,
//        auto tol,
//        size_t max_iters,
//        auto& x,
//        auto& iters,
//        auto& buffer1,
//        auto& buffer2
//    ){
//        update_coefficients(L, v, l1, l2, tol, max_iters, x, iters, buffer1, buffer2);
//    };
//    
//    std::string error;
//    try {
//        cov::fit(pack, update_coefficients_f);
//    } catch(const std::exception& e) {
//        error = e.what(); 
//    }
//    
//    Eigen::SparseMatrix<double> beta_mat;
//    if (pack.n_lmdas > 0) {
//        beta_mat.resize(X.cols(), pack.n_lmdas);
//        for (size_t i = 0; i < pack.n_lmdas; ++i) {
//            beta_mat.col(i) = betas[i];
//        }
//    }
//
//    py::dict d(
//        "rsq"_a=rsq,
//        "strong_beta"_a=pack.strong_beta,
//        "strong_grad"_a=pack.strong_grad,
//        "active_set"_a=*pack.active_set,
//        "active_g1"_a=*pack.active_g1,
//        "active_g2"_a=*pack.active_g2,
//        "active_begins"_a=*pack.active_begins,
//        "active_order"_a=*pack.active_order,
//        "is_active"_a=pack.is_active,
//        "lmdas"_a=lmdas.head(pack.n_lmdas),
//        "betas"_a=beta_mat,
//        "rsqs"_a=pack.rsqs.head(pack.n_lmdas),
//        "n_cds"_a=pack.n_cds,
//        "n_lmdas"_a=pack.n_lmdas,
//        "error"_a=error
//    );
//    return d;
//} 
//
//static py::dict group_elnet_data_newton__(
//    const Eigen::Ref<Eigen::MatrixXd>& X,
//    const Eigen::Ref<Eigen::VectorXi>& groups, 
//    const Eigen::Ref<Eigen::VectorXi>& group_sizes,
//    double alpha, 
//    const Eigen::Ref<Eigen::VectorXd>& penalty,
//    const Eigen::Ref<Eigen::VectorXi>& strong_set, 
//    const std::vector<int>& strong_g1,
//    const std::vector<int>& strong_g2,
//    const Eigen::Ref<Eigen::VectorXi>& strong_begins, 
//    const Eigen::Ref<Eigen::VectorXd>& strong_A_diag,
//    const Eigen::Ref<Eigen::VectorXd>& lmdas, 
//    size_t max_cds,
//    double thr,
//    double cond_0_thresh,
//    double cond_1_thresh,
//    double newton_tol,
//    size_t newton_max_iters,
//    double rsq,
//    Eigen::Ref<Eigen::VectorXd>& strong_beta,
//    Eigen::Ref<Eigen::VectorXd>& strong_grad,
//    std::vector<int> active_set,
//    std::vector<int> active_g1,
//    std::vector<int> active_g2,
//    std::vector<int> active_begins,
//    std::vector<int> active_order,
//    Eigen::Ref<grpglmnet_core::util::vec_type<bool>>& is_active
//)
//{
//    using namespace grpglmnet_core;
//    
//    Eigen::Map<const Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
//    CovCache<Eigen::Map<const Eigen::MatrixXd>, double> A(X_map);
//    util::vec_type<util::sp_vec_type<double, Eigen::ColMajor, int>> betas(lmdas.size());
//    Eigen::VectorXd rsqs(lmdas.size());
//
//    cov::GroupElnetParamPack<
//        std::decay_t<decltype(A)>,
//        double,
//        int,
//        bool
//    > pack(
//        A, groups, group_sizes, alpha, penalty, 
//        strong_set, strong_g1, strong_g2, strong_begins, strong_A_diag,
//        lmdas, max_cds, thr, cond_0_thresh, cond_1_thresh, newton_tol, newton_max_iters, rsq,
//        strong_beta, strong_grad, active_set, active_g1,
//        active_g2, active_begins, active_order, is_active,
//        betas, rsqs, 0, 0
//    );
//    const auto update_coefficients_f = [](
//        const auto& L,
//        const auto& v,
//        auto l1,
//        auto l2,
//        auto tol,
//        size_t max_iters,
//        auto& x,
//        auto& iters,
//        auto& buffer1,
//        auto& buffer2
//    ){
//        grpglmnet_core::newton_solver(L, v, l1, l2, tol, max_iters, x, iters, buffer1, buffer2);
//    };
//
//    std::string error;
//    try {
//        cov::fit(pack, update_coefficients_f);
//    } catch(const std::exception& e) {
//        error = e.what(); 
//    }
//    
//    Eigen::SparseMatrix<double> beta_mat;
//    if (pack.n_lmdas > 0) {
//        beta_mat.resize(X.cols(), pack.n_lmdas);
//        for (size_t i = 0; i < pack.n_lmdas; ++i) {
//            beta_mat.col(i) = betas[i];
//        }
//    }
//
//    py::dict d(
//        "rsq"_a=rsq,
//        "strong_beta"_a=pack.strong_beta,
//        "strong_grad"_a=pack.strong_grad,
//        "active_set"_a=*pack.active_set,
//        "active_g1"_a=*pack.active_g1,
//        "active_g2"_a=*pack.active_g2,
//        "active_begins"_a=*pack.active_begins,
//        "active_order"_a=*pack.active_order,
//        "is_active"_a=pack.is_active,
//        "lmdas"_a=lmdas.head(pack.n_lmdas),
//        "betas"_a=beta_mat,
//        "rsqs"_a=pack.rsqs.head(pack.n_lmdas),
//        "n_cds"_a=pack.n_cds,
//        "n_lmdas"_a=pack.n_lmdas,
//        "error"_a=error
//    );
//    return d;
//} 