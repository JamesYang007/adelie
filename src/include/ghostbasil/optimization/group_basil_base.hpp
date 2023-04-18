#pragma once
#include <Eigen/Core>
#include <Eigen/SVD>
#include <ghostbasil/util/types.hpp>
#include <ghostbasil/util/macros.hpp>
#include <ghostbasil/util/algorithm.hpp>

namespace ghostbasil {
namespace group_lasso {
    
/**
 * @brief Transforms the data matrix X so that each block of data corresponding to a group is 
 * replaced with the principal components. In the process, we also save the l2 norm squared of each column.
 * 
 * @tparam XType        float matrix type.
 * @tparam GroupsType   integer vector type.
 * @tparam GroupSizesType   integer vector type.
 * @tparam DType        float vector type.
 * @param X             data matrix.
 * @param groups        indices to the beginning of each group.
 * @param group_sizes   sizes of each group.
 * @param n_threads     number of threads to parallelize.
 * @param d             l2 norm squared of each column.
 */
template <class XType, class GroupsType, class GroupSizesType, class DType>
GHOSTBASIL_STRONG_INLINE
void transform_data(
    XType& X,
    const GroupsType& groups,
    const GroupSizesType& group_sizes,
    size_t n_threads,
    DType& d
) 
{
#pragma omp parallel for schedule(auto) num_threads(n_threads)
    for (size_t i = 0; i < groups.size(); ++i) {
        const auto gi = groups[i];
        const auto gi_size = group_sizes[i];
        auto Xi = X.block(0, gi, X.rows(), gi_size);
        const auto n = Xi.rows();
        const auto p = Xi.cols();
        const auto m = std::min(n, p);

        Eigen::BDCSVD<Eigen::MatrixXd> solver;
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

GHOSTBASIL_STRONG_INLINE
void screen_edpp(
)
{
}

/**
 * @brief 
 * Append at most max_size number of (first) elements
 * that achieve the maximum absolute value in out.
 * If there are at least max_size number of such elements,
 * exactly max_size will be added.
 * Otherwise, all such elements will be added, but no more.
 * 
 * NOTE: this is EXACTLY the same as the one in basil except the division by alpha logic is a bit different.
 * 
 * @tparam AbsGradType      float vector type.
 * @tparam ValueType        float type.
 * @tparam PenaltyType      float vector type.
 * @tparam ISType           functor: int -> bool.
 * @tparam SSType           int vector type.
 * @param abs_grad          abs_grad[i] is group i's KKT norm value.
 * @param lmda_prev         previous lambda.
 * @param lmda_next         next lambda.
 * @param alpha             elastic net.
 * @param penalty           penalty factor.
 * @param is_strong         functor to check if group i is strong.
 * @param size              number of groups to add if strong-rule not used.
 * @param strong_set        strong set.
 * @param do_strong_rule    true if we should do strong rule.
 */
template <class AbsGradType, class ValueType, class PenaltyType, class ISType, class SSType>
GHOSTBASIL_STRONG_INLINE 
void screen(
    const AbsGradType& abs_grad,
    ValueType lmda_prev,
    ValueType lmda_next,
    ValueType alpha,
    const PenaltyType& penalty,
    const ISType& is_strong,
    size_t size,
    SSType& strong_set,
    bool do_strong_rule
)
{
    using value_t = ValueType;

    assert(strong_set.size() <= abs_grad.size());
    if (!do_strong_rule) {
        size_t rem_size = abs_grad.size() - strong_set.size();
        size_t size_capped = std::min(size, rem_size);
        size_t old_strong_size = strong_set.size();
        strong_set.insert(strong_set.end(), size_capped, 0);
        const auto factor = (alpha <= 1e-16) ? 1e-3 : alpha;
        const auto abs_grad_p = util::vec_type<value_t>::NullaryExpr(
            abs_grad.size(), [&](auto i) {
                return (penalty[i] <= 0) ? 0.0 : abs_grad[i] / penalty[i];
            }
        ) / factor;
        util::k_imax(abs_grad_p, is_strong, size_capped, 
                std::next(strong_set.begin(), old_strong_size));
        return;
    }
    
    const auto strong_rule_lmda = (2 * lmda_next - lmda_prev) * alpha;
    for (size_t i = 0; i < abs_grad.size(); ++i) {
        if (is_strong(i)) continue;
        if (abs_grad[i] > strong_rule_lmda * penalty[i]) {
            strong_set.push_back(i);
        }
    }
}

/**
 * @brief Converts the last valid beta solution into the corresponding strong_beta form
 * and replaces the contents of old_strong_beta.
 * 
 * @tparam OSBType          float vector type.
 * @tparam ValidBetaType    sparse float vector type.
 * @tparam GroupsType       int vector type.
 * @tparam GroupSizesType   int vector type.
 * @tparam SSType           int vector type.
 * @tparam SOType           int vector type.
 * @tparam SBType           int vector type.
 * @param old_strong_beta   old strong beta to replace with equivalent values of beta.
 * @param beta              sparse vector with last valid beta.
 * @param groups            see GroupBasilState.
 * @param group_sizes       see GroupBasilState.
 * @param strong_set        see GroupBasilState. MUST be the old version corresponding to old_strong_beta.
 * @param strong_order      see GroupBasilState. MUST be the old version corresponding to old_strong_beta.
 * @param strong_begins     see GroupBasilState. MUST be the old version corresponding to old_strong_beta.
 */
template <class OSBType, class ValidBetaType, class GroupsType,
          class GroupSizesType, class SSType, class SOType, class SBType>
GHOSTBASIL_STRONG_INLINE
void last_valid_strong_beta(
    OSBType& old_strong_beta,
    const ValidBetaType& beta,
    const GroupsType& groups,
    const GroupSizesType& group_sizes,
    const SSType& strong_set,
    const SOType& strong_order,
    const SBType& strong_begins
)
{
    if (beta.nonZeros() == 0) {
        old_strong_beta.setZero();
        return;
    }
    
    // NOTE: strong_order is non-empty because beta has non-zero elements.
    // All active (non-zero) groups are in strong set, and therefore strong_order is non-empty.

    auto beta_inner = beta.innerIndexPtr();
    auto beta_value = beta.valuePtr();

    size_t outer_pos = 0;
    size_t inner_pos = 0;
    auto so = strong_order[outer_pos];
    auto group = groups[strong_set[so]];
    auto group_size = group_sizes[strong_set[so]];

    for (size_t i = 0; i < beta.nonZeros(); ++i) {
        const auto idx = beta_inner[i];
        const auto val = beta_value[i];

        while (outer_pos < strong_set.size()) {
            const auto begin_ = strong_begins[so] + inner_pos;
            inner_pos = std::min(group_size, idx - group);
            old_strong_beta.segment(
                begin_,
                strong_begins[so] + inner_pos - begin_
            ).array() = 0;

            bool do_break = false;
            if (inner_pos < group_size) {
                old_strong_beta[strong_begins[so] + inner_pos] = val;
                ++inner_pos;
                do_break = true;
            }
            
            if (inner_pos == group_size) {
                inner_pos = 0;
                ++outer_pos;
                if (outer_pos < strong_set.size()) {
                    so = strong_order[outer_pos];
                    group = groups[strong_set[so]];
                    group_size = group_sizes[strong_set[so]];
                }
            }

            if (do_break) break;
        }
    }

    if (outer_pos == strong_set.size()) return;

    old_strong_beta.segment(
        strong_begins[so] + inner_pos,
        group_size - inner_pos
    ).array() = 0;
    ++outer_pos;
        
    // zero out the rest
    while (outer_pos < strong_set.size()) {
        so = strong_order[outer_pos];
        group_size = group_sizes[strong_set[so]];
        old_strong_beta.segment(
            strong_begins[so],
            group_size
        ).array() = 0;
        ++outer_pos;
    }
}

/**
 * @brief Computes the "max" lambda where it is the smallest lambda such that
 * the solution vector is 0. 
 * 
 * @tparam ValueType        float type.
 * @tparam AbsGradType      float vector type.
 * @tparam PenaltyType      float vector type.
 * @param abs_grad          abs_grad[i] is group i's KKT norm value.
 * @param alpha             elastic net.
 * @param penalty           penalty vector.
 * @return the "max" lambda.
 */
template <class ValueType, class AbsGradType, class PenaltyType>
GHOSTBASIL_STRONG_INLINE
auto lambda_max(
    const AbsGradType& abs_grad,
    ValueType alpha,
    const PenaltyType& penalty
)
{
    using value_t = ValueType;
    using vec_value_t = util::vec_type<value_t>;
    const auto factor = (alpha <= 1e-16) ? 1e-3 : alpha;
    return vec_value_t::NullaryExpr(
        abs_grad.size(), [&](auto i) {
            return (penalty[i] <= 0.0) ? 0.0 : abs_grad[i] / penalty[i];
        }
    ).maxCoeff() / factor;
}

/**
 * @brief Generates the sequence of lambdas. EXACTLY the same as in basil.
 * 
 * @tparam ValueType 
 * @tparam OutType 
 * @param max_n_lambdas 
 * @param min_ratio 
 * @param lmda_max 
 * @param out 
 */
template <class ValueType, class OutType>
GHOSTBASIL_STRONG_INLINE
void generate_lambdas(
    size_t max_n_lambdas,
    ValueType min_ratio,
    ValueType lmda_max,
    OutType& out
)
{
    using value_t = ValueType;
    using vec_value_t = util::vec_type<value_t>;

    // lmda_seq = [l_max, l_max * f, l_max * f^2, ..., l_max * f^(max_n_lambdas-1)]
    // l_max is the smallest lambda such that the penalized features (penalty > 0)
    // have 0 coefficients (assuming alpha > 0). The logic is still coherent when alpha = 0.
    auto log_factor = std::log(min_ratio) * static_cast<value_t>(1.0)/(max_n_lambdas-1);
    out.array() = lmda_max * (
        log_factor * vec_value_t::LinSpaced(max_n_lambdas, 0, max_n_lambdas-1)
    ).array().exp();
}


template <class XType, class GroupsType, class GroupSizesType,
          class BetasType>
GHOSTBASIL_STRONG_INLINE 
void untransform_solutions(
    const XType& X,
    const GroupsType& groups,
    const GroupSizesType& group_sizes,
    BetasType& betas,
    size_t n_threads
)
{
    using value_t = typename std::decay_t<XType>::Scalar;
    using mat_t = util::mat_type<value_t>;
    using vec_t = util::vec_type<value_t>;
    
    if (betas.size() <= 0) return;

    // Parallelize over groups so that we can save time on SVD.
    vec_t transformed_beta(X.cols());
#pragma omp parallel for schedule(auto) num_threads(n_threads)
    for (size_t j = 0; j < groups.size(); ++j) {
        const auto gj = groups[j];
        const auto gj_size = group_sizes[j];
        auto trans_beta_j = transformed_beta.segment(gj, gj_size);
        
        // Optimization: beta non-zero entries is always increasing (active set is ever-active).
        // So, if the last beta vector is not active in current group, no point of SVD'ing.
        const auto last_beta = betas.back();
        const auto inner = last_beta.innerIndexPtr();
        const auto value = last_beta.valuePtr();
        const auto nzn = last_beta.nonZeros();
        if (nzn == 0) continue;
        const auto idx = std::lower_bound(
            inner,
            inner + nzn,
            gj
        ) - inner;
        if (idx == nzn || inner[idx] >= gj + gj_size) continue;

        Eigen::BDCSVD<mat_t> solver(X.block(0, gj, X.rows(), gj_size), Eigen::ComputeFullV);

        for (size_t i = 0; i < betas.size(); ++i) {
            auto& beta_i = betas[i];
            const auto inner_i = beta_i.innerIndexPtr();
            const auto value_i = beta_i.valuePtr();
            const auto nzn_i = beta_i.nonZeros();

            if (nzn_i == 0) continue;

            const auto idx = std::lower_bound(
                inner_i,
                inner_i + nzn_i,
                gj
            ) - inner_i;
            
            if (idx == nzn_i || inner_i[idx] >= gj + gj_size) continue;
            
            // guaranteed that gj <= inner[idx] < gj + gj_size
            // By construction of beta_i, it only contains active group coefficients.
            // So, we must have that gj == inner[idx].
            // Moreover, value[idx : idx + gj_size] should be the beta_i's jth group coefficients.
            if (inner_i[idx] != gj) throw std::runtime_error("Index of non-zero block does not start at expected position. This is an indication that there is a bug! Please report this.");
            if (idx + gj_size > nzn_i) throw std::runtime_error("Index of non-zero block are not fully active. This is an indication that there is a bug! Please report this.");
            if (inner_i[idx + gj_size - 1] != gj + gj_size - 1) throw std::runtime_error("Index of non-zero block does not end at expected position. This is an indication that there is a bug! Please report this.");

            Eigen::Map<vec_t> beta_i_j_map(
                value_i + idx, gj_size
            );
            trans_beta_j.noalias() = solver.matrixV() * beta_i_j_map;
            beta_i_j_map = trans_beta_j;
        }
    }
    
    for (size_t i = 0; i < betas.size(); ++i) {
        auto& beta_i = betas[i];
        beta_i.prune(0, 1e-14);
    }
}

} // namespace group_lasso
} // namespace ghostbasil