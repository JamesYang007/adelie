#pragma once
#include <Eigen/SVD>
#include <adelie_core/util/types.hpp>
#include <adelie_core/util/macros.hpp>

namespace adelie_core {
namespace grpnet {

template <class XType, class GroupsType, class GroupSizesType, class DType>
ADELIE_CORE_STRONG_INLINE
void transform_data(
    XType& X,
    const GroupsType& groups,
    const GroupSizesType& group_sizes,
    size_t n_threads,
    DType& d
) 
{
    using value_t = typename std::decay_t<XType>::Scalar;
#pragma omp parallel for schedule(auto) num_threads(n_threads)
    for (size_t i = 0; i < groups.size(); ++i) {
        const auto gi = groups[i];
        const auto gi_size = group_sizes[i];
        auto Xi = X.block(0, gi, X.rows(), gi_size);
        const auto n = Xi.rows();
        const auto p = Xi.cols();
        const auto m = std::min(n, p);

        Eigen::BDCSVD<util::colmat_type<value_t>> solver;
        solver.compute(Xi, Eigen::ComputeThinU);
        const auto& U = solver.matrixU();
        const auto& D = solver.singularValues();

        Xi.block(0, m, n, p-m).setZero();
        auto Xi_sub = Xi.block(0, 0, n, m);
        Xi_sub.array() = U.array().rowwise() * D.transpose().array();
        d.segment(gi, m) = D.array().square();
        d.segment(gi + m, p-m).setZero();
    }
}

//template <class XType, class GroupsType, class GroupSizesType, class BetasType>
//ADELIE_CORE_STRONG_INLINE
//void untransform_solutions(
//    const XType& X,
//    const GroupsType& groups,
//    const GroupSizesType& group_sizes,
//    BetasType& betas,
//    size_t n_threads
//)
//{
//    using value_t = typename std::decay_t<XType>::Scalar;
//    using mat_t = util::mat_type<value_t>;
//    using vec_t = util::vec_type<value_t>;
//    
//    if (betas.size() <= 0) return;
//
//    // Parallelize over groups so that we can save time on SVD.
//    vec_t transformed_beta(X.cols());
//#pragma omp parallel for schedule(auto) num_threads(n_threads)
//    for (size_t j = 0; j < groups.size(); ++j) {
//        const auto gj = groups[j];
//        const auto gj_size = group_sizes[j];
//        auto trans_beta_j = transformed_beta.segment(gj, gj_size);
//        
//        // Optimization: beta non-zero entries is always increasing (active set is ever-active).
//        // So, if the last beta vector is not active in current group, no point of SVD'ing.
//        const auto last_beta = betas.back();
//        const auto inner = last_beta.innerIndexPtr();
//        const auto value = last_beta.valuePtr();
//        const auto nzn = last_beta.nonZeros();
//        if (nzn == 0) continue;
//        const auto idx = std::lower_bound(
//            inner,
//            inner + nzn,
//            gj
//        ) - inner;
//        if (idx == nzn || inner[idx] >= gj + gj_size) continue;
//
//        Eigen::BDCSVD<mat_t> solver(X.block(0, gj, X.rows(), gj_size), Eigen::ComputeFullV);
//
//        for (size_t i = 0; i < betas.size(); ++i) {
//            auto& beta_i = betas[i];
//            const auto inner_i = beta_i.innerIndexPtr();
//            const auto value_i = beta_i.valuePtr();
//            const auto nzn_i = beta_i.nonZeros();
//
//            if (nzn_i == 0) continue;
//
//            const auto idx = std::lower_bound(
//                inner_i,
//                inner_i + nzn_i,
//                gj
//            ) - inner_i;
//            
//            if (idx == nzn_i || inner_i[idx] >= gj + gj_size) continue;
//            
//            // guaranteed that gj <= inner[idx] < gj + gj_size
//            // By construction of beta_i, it only contains active group coefficients.
//            // So, we must have that gj == inner[idx].
//            // Moreover, value[idx : idx + gj_size] should be the beta_i's jth group coefficients.
//            if (inner_i[idx] != gj) throw std::runtime_error("Index of non-zero block does not start at expected position. This is an indication that there is a bug! Please report this.");
//            if (idx + gj_size > nzn_i) throw std::runtime_error("Index of non-zero block are not fully active. This is an indication that there is a bug! Please report this.");
//            if (inner_i[idx + gj_size - 1] != gj + gj_size - 1) throw std::runtime_error("Index of non-zero block does not end at expected position. This is an indication that there is a bug! Please report this.");
//
//            Eigen::Map<vec_t> beta_i_j_map(
//                value_i + idx, gj_size
//            );
//            trans_beta_j.noalias() = solver.matrixV() * beta_i_j_map;
//            beta_i_j_map = trans_beta_j;
//        }
//    }
//}

template <class ValueType, class AbsGradType, class PenaltyType>
ADELIE_CORE_STRONG_INLINE
auto lambda_max(
    const AbsGradType& abs_grad,
    ValueType alpha,
    const PenaltyType& penalty
)
{
    using value_t = ValueType;
    using vec_value_t = util::rowvec_type<value_t>;
    const auto factor = (alpha <= 0) ? 1e-3 : alpha;
    return vec_value_t::NullaryExpr(
        abs_grad.size(), [&](auto i) {
            return (penalty[i] <= 0.0) ? 0.0 : abs_grad[i] / penalty[i];
        }
    ).maxCoeff() / factor;
}

template <class ValueType, class OutType>
ADELIE_CORE_STRONG_INLINE
void create_lambdas(
    size_t max_n_lambdas,
    ValueType min_ratio,
    ValueType lmda_max,
    OutType& out
)
{
    using value_t = ValueType;
    using vec_value_t = util::rowvec_type<value_t>;

    // lmda_seq = [l_max, l_max * f, l_max * f^2, ..., l_max * f^(max_n_lambdas-1)]
    // l_max is the smallest lambda such that the penalized features (penalty > 0)
    // have 0 coefficients (assuming alpha > 0). The logic is still coherent when alpha = 0.
    auto log_factor = std::log(min_ratio) * static_cast<value_t>(1.0)/(max_n_lambdas-1);
    out = lmda_max * (
        log_factor * vec_value_t::LinSpaced(max_n_lambdas, 0, max_n_lambdas-1)
    ).exp();
}

} // namespace grpnet
} // namespace adelie_core