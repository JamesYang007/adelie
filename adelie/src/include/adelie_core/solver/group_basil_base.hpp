#pragma once
#include <Eigen/Core>
#include <Eigen/SVD>
#include <adelie_core/util/types.hpp>
#include <adelie_core/util/macros.hpp>
#include <adelie_core/util/algorithm.hpp>
#include <omp.h>

namespace adelie_core {

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