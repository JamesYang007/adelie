#pragma once
#include <adelie_core/util/types.hpp>

namespace adelie_core {
namespace optimization {

template <class MatrixType,
          class ValueType=typename std::decay_t<MatrixType>::value_t>
struct StateNNQPBasic
{
    using matrix_t = MatrixType;
    using value_t = ValueType;
    using vec_value_t = util::rowvec_type<value_t>;
    using colmat_value_t = util::colmat_type<value_t>;
    using map_vec_value_t = Eigen::Map<vec_value_t>;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;
    using map_ccolmat_value_t = Eigen::Map<const colmat_value_t>;

    const value_t rho;
    const map_ccolmat_value_t Sigma;
    const map_cvec_value_t quad_diag;
    const map_cvec_value_t v;

    const size_t max_iters;
    const value_t tol;

    size_t iters = 0;
    matrix_t* A;
    map_vec_value_t x;      
    map_vec_value_t resid;  // A^T x  

    double time_elapsed = 0;

    StateNNQPBasic(
        matrix_t& A,
        value_t rho,
        const Eigen::Ref<const colmat_value_t>& Sigma,
        const Eigen::Ref<const vec_value_t>& quad_diag,
        const Eigen::Ref<const vec_value_t>& v,
        size_t max_iters,
        value_t tol,
        Eigen::Ref<vec_value_t> x,
        Eigen::Ref<vec_value_t> resid
    ):
        rho(rho),
        Sigma(Sigma.data(), Sigma.rows(), Sigma.cols()),
        v(v.data(), v.size()),
        quad_diag(quad_diag.data(), quad_diag.size()),
        max_iters(max_iters),
        tol(tol),
        A(&A),
        x(x.data(), x.size()),
        resid(resid.data(), resid.size())
    {}
};

template <class StateType, class BufferType>
void nnqp_basic(
    StateType& state,
    BufferType& buff
)
{
    using state_t = std::decay_t<StateType>;
    using value_t = typename state_t::value_t;
    using vec_value_t = typename state_t::vec_value_t;

    const auto rho = state.rho;
    const auto& Sigma = state.Sigma;
    const auto& quad_diag = state.quad_diag;
    const auto& v = state.v;
    const auto max_iters = state.max_iters;
    const auto tol = state.tol;
    auto A = state.A;
    auto& iters = state.iters;
    auto& x = state.x;
    auto& resid = state.resid;

    const auto n = quad_diag.size();
    const auto d = Sigma.rows();

    Eigen::Map<vec_value_t> Sigma_resid(
        buff.data(), d
    );

    iters = 0;
    while (iters < max_iters) {
        value_t convg_measure = 0;
        ++iters;
        for (int k = 0; k < n; ++k) {
            const auto v_k = v[k];
            const auto H_kk = quad_diag[k];
            auto& x_k = x[k];
            Sigma_resid = resid.matrix() * Sigma;
            const auto gk = v_k - rho * A->rmul(k, Sigma_resid);
            const auto x_k_old = x_k;
            x_k = std::max<value_t>(x_k + gk / H_kk, 0.0);
            if (x_k == x_k_old) continue;
            const auto del = x_k - x_k_old;
            convg_measure = std::max<value_t>(convg_measure, H_kk * del * del);
            A->rtmul(k, del, resid);
        }
        if (convg_measure < tol) break;
    }
}

} // namespace optimization
} // namespace adelie_core