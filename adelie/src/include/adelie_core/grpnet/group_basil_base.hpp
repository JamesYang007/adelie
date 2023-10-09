#pragma once
#include <Eigen/Core>
#include <Eigen/SVD>
#include <adelie_core/util/types.hpp>
#include <adelie_core/util/macros.hpp>
#include <adelie_core/util/algorithm.hpp>
#include <omp.h>

namespace adelie_core {
    
template <class XType, class GroupsType, class GroupSizesType, 
          class XGNType, class PenaltyType, class GradType, 
          class Resid0Type, class ResidType, class ValueType, class V1Type,
          class IESType, class ESSType>
ADELIEE
void screen_edpp(
    const XType& X,
    const GroupsType& groups,
    const GroupSizesType& group_sizes,
    const XGNType& X_group_norms,
    ValueType alpha,
    const PenaltyType& penalty,
    const GradType& grad,
    const Resid0Type& resid_0,
    const ResidType& resid,
    ValueType lmda_curr,
    ValueType lmda_prev,
    bool is_lmda_prev_max,
    const V1Type& v1_0,
    IESType is_edpp_safe,
    size_t n_threads,
    ESSType& edpp_safe_set
)
{
    if (alpha != 1) return;

    Eigen::VectorXd v1 = (
        (is_lmda_prev_max) ?
        v1_0 :
        (resid_0 - resid) / lmda_prev
    );
    Eigen::VectorXd v2 = resid_0 / lmda_curr - resid / lmda_prev;
    Eigen::VectorXd v2_perp = v2 - ((v1.dot(v2)) / v1.squaredNorm()) * v1;
    const auto v2_perp_norm = v2_perp.norm();
    Eigen::VectorXd buffer(X.cols());    

    std::vector<std::vector<int>> edpp_new_safe_threads(n_threads);

#pragma omp parallel for schedule(auto) num_threads(n_threads)
    for (size_t i = 0; i < groups.size(); ++i) {
        if (is_edpp_safe(i)) continue;
        const auto g = groups[i];
        const auto gs = group_sizes[i];
        const auto Xg = X.block(0, g, X.rows(), gs);
        const auto grad_g = grad.segment(g, gs);
        auto buff_g = buffer.segment(g, gs);
        buff_g.noalias() = 0.5 * (Xg.transpose() * v2_perp);
        buff_g += grad_g / lmda_prev;
        const auto buff_g_norm = buff_g.norm();
        if (buff_g_norm >= penalty[i] - 0.5 * v2_perp_norm * X_group_norms[i]) {
            const auto thread_i = omp_get_thread_num();
            edpp_new_safe_threads[thread_i].push_back(i);
        }
    }    
    
    for (const auto& edpp_new_safe : edpp_new_safe_threads) {
        edpp_safe_set.insert(edpp_safe_set.end(), edpp_new_safe.begin(), edpp_new_safe.end());
    }
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
ADELIEE 
void screen(
    const AbsGradType& abs_grad,
    ValueType lmda_prev,
    ValueType lmda_next,
    ValueType alpha,
    const PenaltyType& penalty,
    const ISType& is_strong,
    size_t size,
    size_t rem_size,
    SSType& strong_set,
    bool do_strong_rule
)
{
    using value_t = ValueType;

    assert(strong_set.size() <= abs_grad.size());
    if (!do_strong_rule) {
        size_t size_capped = std::min(size, rem_size);
        size_t old_strong_size = strong_set.size();
        strong_set.insert(strong_set.end(), size_capped, -1);
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
ADELIEE
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



} // namespace adelie_core