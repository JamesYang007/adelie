#pragma once
#include <adelie_core/glm/glm_cox.hpp>

namespace adelie_core {
namespace glm {
namespace cox {

/**
 * Computes the partial sums based on the thresholds given by ``t`` and time-points ``s``.
 * 
 *      out[i] = \sum_{k=1}^n v[k] 1_{s[k] <= t[i]}
 * 
 * @param v     (n,) array of values to sum in s-order.
 * @param s     (n,) array of increasing time-points.
 * @param t     (m,) array of increasing thresholds.
 * @param out   (m+1,) array of sums with padding out[0] == 0 in t-order.
 */
template <class VType, class SType, class TType, class OutType>
void _partial_sum_fwd(
    const VType& v, 
    const SType& s,
    const TType& t,
    OutType& out
)
{
    const auto n = s.size();
    const auto m = t.size();
    out[0] = 0;

    if (m == 0) return;
    if (n == 0) {
        out.setZero();
        return;
    }

    int k_begin = 0;    // begin of s/v array
    int i_begin = 0;    // begin block with same t_i
    int i_end = 0;      // end block with same t_i

    while (i_begin < m) {
        const auto ti = t[i_begin];
        auto curr_sum = out[i_begin];
        // accumulate more terms while s_k <= t_i
        for (; k_begin < n && s[k_begin] <= ti; ++k_begin) curr_sum += v(k_begin);
        // save current sum for all positions with time == t_i
        for (; i_end < m && t[i_end] == ti; ++i_end) out[i_end+1] = curr_sum;
        i_begin = i_end;
        if (k_begin >= n) break;
    }

    for (; i_begin < m; ++i_begin) out[i_begin+1] = out[i_begin];
}

/**
 * Computes the partial sums based on the thresholds given by ``t`` and time-points ``s``.
 * 
 *      out[i] = \sum_{k=1}^n v[k] 1_{s[k] >= t[i]}
 * 
 * @param v     (n,) array of values to sum in s-order.
 * @param s     (n,) array of increasing time-points.
 * @param t     (m,) array of increasing thresholds.
 * @param out   (m+1,) array of sums with padding out[m] == 0 in t-order.
 */
template <class VType, class SType, class TType, class OutType>
void _partial_sum_bwd(
    const VType& v, 
    const SType& s,
    const TType& t,
    OutType& out
)
{
    const auto n = s.size();
    const auto m = t.size();
    out[m] = 0;

    if (m == 0) return;
    if (n == 0) {
        out.setZero();
        return;
    }

    int k_begin = n-1;    // begin of s/v array
    int i_begin = m-1;    // begin block with same t_i
    int i_end = m-1;      // end block with same t_i

    while (i_begin >= 0) {
        const auto ti = t[i_begin];
        auto curr_sum = out[i_begin+1];
        // accumulate more terms while s_k >= t_i
        for (; k_begin >= 0 && s[k_begin] >= ti; --k_begin) curr_sum += v(k_begin);
        // save current sum for all positions with time == t_i
        for (; i_end >= 0 && t[i_end] == ti; --i_end) out[i_end] = curr_sum;
        i_begin = i_end;
        if (k_begin < 0) break;
    }

    for (; i_begin >= 0; --i_begin) out[i_begin] = out[i_begin+1];
}

/**
 * Computes the sum of values within each at-risk set:
 * 
 *      out[i] = (
 *          \sum_{k=1}^n a_t[k] 1_{t[k] >= u[i]} -
 *          \sum_{k=1}^n a_s[k] 1_{s[k] >= u[i]}
 *      )
 * 
 * @param a_s       (n,) array of values to sum in s-order.
 * @param a_t       (n,) array of values to sum in t-order. 
 * @param s         (n,) array of increasing time-points.
 * @param t         (n,) array of increasing time-points.
 * @param u         (m,) array of increasing thresholds.
 * @param out       (m,) array which is out1 - out2, without the padding in u-order.
 * @param out1      (m+1,) array of first term with padding out[m] == 0 in u-order.
 * @param out2      (m+1,) array of second term with padding out[m] == 0 in u-order.
 */
template <class ASType, class ATType, class SType, class TType, 
          class UType, class OutType, class Out1Type, class Out2Type>
void _at_risk_sum(
    const ASType& a_s,
    const ATType& a_t,
    const SType& s,
    const TType& t,
    const UType& u,
    OutType& out,
    Out1Type& out1,
    Out2Type& out2
)
{
    _partial_sum_bwd(a_t, t, u, out1);
    _partial_sum_bwd(a_s, s, u, out2);
    const auto m = out.size();
    out = out1.head(m) - out2.head(m);
}

/**
 * Computes the sum of values within each event tie and non-zero weights.
 * 
 *      out[i] = (
 *          status[i] * (w[i] != 0) * 
 *          \sum_{k=1}^n a_k * 1_{t[k] = t[i], status[k] = 1, w[k] != 0}
 *      )
 * 
 * @param a         (n,) array of values in t-order.
 * @param t         (n,) array of increasing time-points.
 * @param status    (n,) array of event indicators in t-order.
 * @param w         (n,) array of weights in t-order.
 * @param out       (n,) array of sums in t-order.
 */
template <class AType, class TType, class StatusType, class WType, class OutType>
void _nnz_event_ties_sum(
    const AType& a,
    const TType& t,
    const StatusType& status,
    const WType& w,
    OutType& out
)
{
    using value_t = typename std::decay_t<WType>::Scalar;
    const auto n = w.size();
    int i_begin = 0;
    while (i_begin < n) {
        const auto ti = t[i_begin];
        int i_end = i_begin;
        value_t sum = 0;
        for (; i_end < n && t[i_end] == ti; ++i_end) {
            const auto indic = status[i_end] * (w[i_end] != 0);
            sum += a(i_end) * indic;
        }
        for (int j = i_begin; j < i_end; ++j) {
            out[j] = status[j] * (w[j] != 0) * sum;
        }
        i_begin = i_end;
    }
}

/**
 * Computes the scale for tie-breaking method.
 * 
 *      out[i] = status[i] * (w[i] != 0) * k_i / s[i]  
 * 
 * where k_i is an ordering (0-indexed) among the event ties with positive weights,
 * and s[i] is the number of event ties with positive weights at t[i]
 * (output of _nnz_event_ties_size()).
 * If Breslow method, then k_i = 0 for all i.
 * 
 * @param t         (n,) array of increasing time-points.
 * @param status    (n,) array of event indicators in t-order.
 * @param w         (n,) array of weights in t-order.
 * @param tie_method  tie-breaking method.
 * @param out       (n,) array of scales in t-order.
 */
template <class TType, class StatusType, class WType, class OutType>
void _scale(
    const TType& t,
    const StatusType& status,
    const WType& w,
    util::tie_method_type tie_method,
    OutType& out
)
{
    using value_t = typename std::decay_t<TType>::Scalar;
    if (tie_method == util::tie_method_type::_breslow) {
        out.setZero();
        return;
    }
    const auto n = t.size();
    int i_begin = 0;
    while (i_begin < n) {
        auto ti = t[i_begin];
        int i_end = i_begin;
        int size = 0;
        for (; i_end < n && t[i_end] == ti; ++i_end) {
            const auto indic = status[i_end] * (w[i_end] != 0);
            out[i_end] = size * indic;
            size += indic;
        }
        if (size > 1) {
            Eigen::Map<util::rowvec_type<value_t>>(
                out.data() + i_begin,
                i_end - i_begin
            ) /= size;
        }
        i_begin = i_end;
    }
}

} // namespace cox

ADELIE_CORE_GLM_COX_PACK_TP
auto
ADELIE_CORE_GLM_COX_PACK::init_order(
    const Eigen::Ref<const vec_value_t>& x
)
{
    vec_index_t x_order = vec_index_t::LinSpaced(x.size(), 0, x.size()-1);
    std::sort(
        x_order.data(), 
        x_order.data() + x_order.size(),
        [&](auto i, auto j) { return x[i] < x[j]; }
    );
    return x_order;
}

ADELIE_CORE_GLM_COX_PACK_TP
auto
ADELIE_CORE_GLM_COX_PACK::init_in_order(
    const Eigen::Ref<const vec_value_t>& x,
    const Eigen::Ref<const vec_index_t>& order
)
{
    vec_value_t x_sorted(x.size());
    for (int i = 0; i < order.size(); ++i) {
        x_sorted[i] = x[order[i]];
    }
    return x_sorted;
} 

ADELIE_CORE_GLM_COX_PACK_TP
auto
ADELIE_CORE_GLM_COX_PACK::init_weights_size_to(
    const Eigen::Ref<const vec_value_t>& stop_to,
    const Eigen::Ref<const vec_value_t>& status_to,
    const Eigen::Ref<const vec_value_t>& weights_to
)
{
    vec_value_t weights_size_to(stop_to.size());
    cox::_nnz_event_ties_sum(
        vec_value_t::Ones(stop_to.size()),
        stop_to,
        status_to,
        weights_to,
        weights_size_to
    );
    return weights_size_to;
}

ADELIE_CORE_GLM_COX_PACK_TP
auto
ADELIE_CORE_GLM_COX_PACK::init_weights_mean_to(
    const Eigen::Ref<const vec_value_t>& stop_to,
    const Eigen::Ref<const vec_value_t>& status_to,
    const Eigen::Ref<const vec_value_t>& weights_to,
    const Eigen::Ref<const vec_value_t>& weights_size_to
)
{
    const auto n = stop_to.size();
    vec_value_t weights_mean_to(n);
    cox::_nnz_event_ties_sum(
        weights_to,
        stop_to,
        status_to,
        weights_to,
        weights_mean_to
    );
    for (int i = 0; i < n; ++i) {
        if (!status_to[i] || !weights_to[i]) continue;
        weights_mean_to[i] /= weights_size_to[i];
    }
    return weights_mean_to;
}

ADELIE_CORE_GLM_COX_PACK_TP
auto
ADELIE_CORE_GLM_COX_PACK::init_scale_to(
    const Eigen::Ref<const vec_value_t>& stop_to,
    const Eigen::Ref<const vec_value_t>& status_to,
    const Eigen::Ref<const vec_value_t>& weights_to,
    util::tie_method_type tie_method
)
{
    vec_value_t scale_to(stop_to.size());
    cox::_scale(
        stop_to,
        status_to,
        weights_to,
        tie_method,
        scale_to
    );
    return scale_to;
}

ADELIE_CORE_GLM_COX_PACK_TP
ADELIE_CORE_GLM_COX_PACK::GlmCoxPack(
    const Eigen::Ref<const vec_value_t>& start,
    const Eigen::Ref<const vec_value_t>& stop,
    const Eigen::Ref<const vec_value_t>& status,
    const Eigen::Ref<const vec_value_t>& weights,
    const std::string& tie_method_str
):
    tie_method(util::convert_tie_method(tie_method_str)),
    start(start.data(), start.size()),
    stop(stop.data(), stop.size()),
    status(status.data(), status.size()),
    weights(weights.data(), weights.size()),
    start_order(init_order(start)),
    start_so(init_in_order(start, start_order)),
    stop_order(init_order(stop)),
    stop_to(init_in_order(stop, stop_order)),
    status_to(init_in_order(status, stop_order)),
    weights_to(init_in_order(weights, stop_order)),
    weights_size_to(init_weights_size_to(
        stop_to, status_to, weights_to
    )),
    weights_mean_to(init_weights_mean_to(
        stop_to, status_to, weights_to, weights_size_to
    )),
    scale_to(init_scale_to(
        stop_to, status_to, weights_to, tie_method
    )),
    buffer(5 * (start.size() + 1))
{}

ADELIE_CORE_GLM_COX_PACK_TP
void
ADELIE_CORE_GLM_COX_PACK::gradient(
    const Eigen::Ref<const vec_value_t>& eta,
    Eigen::Ref<vec_value_t> grad
) 
{
    const auto n = eta.size();
    Eigen::Map<vec_value_t> z(buffer.data(), n);
    z = weights * eta.exp();

    // compute b_k
    Eigen::Map<vec_value_t> risk_sum_to(buffer.data() + n, n);
    Eigen::Map<vec_value_t> risk_sum1_to(buffer.data() + 2 * n, n + 1);
    Eigen::Map<vec_value_t> risk_sum2_to(buffer.data() + 3 * n + 1, n + 1);
    cox::_at_risk_sum(
        [&](auto i) { return z[start_order[i]]; },
        [&](auto i) { return z[stop_order[i]]; },
        start_so,
        stop_to,
        stop_to,
        risk_sum_to,
        risk_sum1_to,
        risk_sum2_to
    );
    Eigen::Map<vec_value_t> ties_risk_sum_to(buffer.data() + 2 * n, n);
    cox::_nnz_event_ties_sum(
        [&](auto i) { return z[stop_order[i]]; },
        stop_to,
        status_to,
        weights_to,
        ties_risk_sum_to
    );
    Eigen::Map<vec_value_t> risk_total_to(buffer.data() + 3 * n, n);
    risk_total_to = risk_sum_to - scale_to * ties_risk_sum_to;

    // compute gradient scales
    Eigen::Map<vec_value_t> _v_to(buffer.data() + n, n);
    _v_to = status_to * weights_mean_to / (
        risk_total_to + ((status_to == 0) || (weights_mean_to == 0)).template cast<value_t>()
    );
    Eigen::Map<vec_value_t> _gs1_to(buffer.data() + 2 * n, n + 1);
    cox::_partial_sum_fwd(_v_to, stop_to, stop_to, _gs1_to);
    Eigen::Map<vec_value_t> _gs2_so(buffer.data() + 3 * n + 1, n + 1);
    cox::_partial_sum_fwd(_v_to, stop_to, start_so, _gs2_so);
    Eigen::Map<vec_value_t> _gs3_to(buffer.data() + 4 * n + 2, n);
    _v_to *= scale_to;
    cox::_nnz_event_ties_sum(_v_to, stop_to, status_to, weights_to, _gs3_to);

    // compute gradient
    for (int i = 0; i < n; ++i) grad[stop_order[i]] = _gs1_to[i+1] - _gs3_to[i];
    for (int i = 0; i < n; ++i) grad[start_order[i]] -= _gs2_so[i+1];
    grad = weights * status - grad * z;
}

ADELIE_CORE_GLM_COX_PACK_TP
void
ADELIE_CORE_GLM_COX_PACK::hessian(
    const Eigen::Ref<const vec_value_t>& eta,
    const Eigen::Ref<const vec_value_t>& grad,
    Eigen::Ref<vec_value_t> hess
) 
{
    const auto n = eta.size();
    Eigen::Map<vec_value_t> z(buffer.data(), n);
    z = weights * eta.exp();

    // compute b_k
    Eigen::Map<vec_value_t> risk_sum_to(buffer.data() + n, n);
    Eigen::Map<vec_value_t> risk_sum1_to(buffer.data() + 2 * n, n + 1);
    Eigen::Map<vec_value_t> risk_sum2_to(buffer.data() + 3 * n + 1, n + 1);
    cox::_at_risk_sum(
        [&](auto i) { return z[start_order[i]]; },
        [&](auto i) { return z[stop_order[i]]; },
        start_so,
        stop_to,
        stop_to,
        risk_sum_to,
        risk_sum1_to,
        risk_sum2_to
    );
    Eigen::Map<vec_value_t> ties_risk_sum_to(buffer.data() + 2 * n, n);
    cox::_nnz_event_ties_sum(
        [&](auto i) { return z[stop_order[i]]; },
        stop_to,
        status_to,
        weights_to,
        ties_risk_sum_to
    );
    Eigen::Map<vec_value_t> risk_total_to(buffer.data() + 3 * n, n);
    risk_total_to = risk_sum_to - scale_to * ties_risk_sum_to;

    // compute hessian scales
    Eigen::Map<vec_value_t> _v_to(buffer.data() + n, n);
    _v_to = status_to * weights_mean_to / (
        risk_total_to.square() + ((status_to == 0) || (weights_mean_to == 0)).template cast<value_t>()
    );
    Eigen::Map<vec_value_t> _hs1_to(buffer.data() + 2 * n, n + 1);
    cox::_partial_sum_fwd(_v_to, stop_to, stop_to, _hs1_to);
    Eigen::Map<vec_value_t> _hs2_so(buffer.data() + 3 * n + 1, n + 1);
    cox::_partial_sum_fwd(_v_to, stop_to, start_so, _hs2_so);
    Eigen::Map<vec_value_t> _hs3_to(buffer.data() + 4 * n + 2, n);
    _v_to *= scale_to * (2 - scale_to);
    cox::_nnz_event_ties_sum(_v_to, stop_to, status_to, weights_to, _hs3_to);

    // compute hessian
    for (int i = 0; i < n; ++i) hess[stop_order[i]] = _hs1_to[i+1] - _hs3_to[i];
    for (int i = 0; i < n; ++i) hess[start_order[i]] -= _hs2_so[i+1];
    hess = weights * status - grad - hess * z.square();
}

ADELIE_CORE_GLM_COX_PACK_TP
typename ADELIE_CORE_GLM_COX_PACK::value_t
ADELIE_CORE_GLM_COX_PACK::loss(
    const Eigen::Ref<const vec_value_t>& eta
) 
{
    constexpr auto neg_max = -std::numeric_limits<value_t>::max();
    const auto n = eta.size();
    const auto eta_max = eta.maxCoeff();
    Eigen::Map<vec_value_t> z(buffer.data(), n);
    z = weights * (eta-eta_max).exp();
    Eigen::Map<vec_value_t> risk_sum_to(buffer.data() + n, n);
    Eigen::Map<vec_value_t> risk_sum1_to(buffer.data() + 2 * n, n + 1);
    Eigen::Map<vec_value_t> risk_sum2_to(buffer.data() + 3 * n + 1, n + 1);
    cox::_at_risk_sum(
        [&](auto i) { return z[start_order[i]]; },
        [&](auto i) { return z[stop_order[i]]; },
        start_so,
        stop_to,
        stop_to,
        risk_sum_to,
        risk_sum1_to,
        risk_sum2_to
    );
    Eigen::Map<vec_value_t> ties_risk_sum_to(buffer.data() + 2 * n, n);
    cox::_nnz_event_ties_sum(
        [&](auto i) { return z[stop_order[i]]; },
        stop_to,
        status_to,
        weights_to,
        ties_risk_sum_to
    );
    return (
        - (status * weights * (eta-eta_max)).sum()
        + (status_to * weights_mean_to * 
            (risk_sum_to - scale_to * ties_risk_sum_to).max(0).log().max(neg_max)
        ).sum()
    );
}

ADELIE_CORE_GLM_COX_PACK_TP
typename ADELIE_CORE_GLM_COX_PACK::value_t
ADELIE_CORE_GLM_COX_PACK::loss_full() 
{
    const constexpr auto most_neg = -std::numeric_limits<value_t>::max();
    return (
        weights_mean_to * status_to * 
        (weights_size_to * weights_mean_to * (1 - scale_to)).log().max(most_neg)
    ).sum();
}

ADELIE_CORE_GLM_COX_TP
auto
ADELIE_CORE_GLM_COX::init_strata_outer(
    const Eigen::Ref<const vec_index_t>& strata,
    size_t n_stratas
)
{
    vec_index_t strata_outer(n_stratas + 1);
    strata_outer.setZero();
    for (index_t i = 0; i < strata.size(); ++i) {
        const auto si = strata[i];
        ++strata_outer[si+1];
    }
    for (index_t i = 1; i < strata_outer.size(); ++i) {
        strata_outer[i] += strata_outer[i-1];
    }
    return strata_outer;
}

ADELIE_CORE_GLM_COX_TP
auto
ADELIE_CORE_GLM_COX::init_strata_order(
    const Eigen::Ref<const vec_index_t>& strata
)
{
    const auto n = strata.size();
    vec_index_t order = vec_index_t::LinSpaced(n, 0, n-1);
    std::sort(
        order.data(),
        order.data() + n,
        [&](auto i, auto j) {
            const auto si = strata[i];
            const auto sj = strata[j];
            return (si < sj) || ((si == sj) && (i < j));
        }
    );
    return order;
}

ADELIE_CORE_GLM_COX_TP
void
ADELIE_CORE_GLM_COX::init_in_order(
    const Eigen::Ref<const vec_value_t>& x,
    const Eigen::Ref<const vec_index_t>& order,
    Eigen::Ref<vec_value_t> x_sorted
)
{
    for (int i = 0; i < order.size(); ++i) {
        x_sorted[i] = x[order[i]];
    }
} 

ADELIE_CORE_GLM_COX_TP
auto
ADELIE_CORE_GLM_COX::init_in_order(
    const Eigen::Ref<const vec_value_t>& x,
    const Eigen::Ref<const vec_index_t>& order
)
{
    vec_value_t x_sorted(x.size());
    init_in_order(x, order, x_sorted);
    return x_sorted;
} 

ADELIE_CORE_GLM_COX_TP
void
ADELIE_CORE_GLM_COX::init_from_order(
    const Eigen::Ref<const vec_value_t>& x_sorted,
    const Eigen::Ref<const vec_index_t>& order,
    Eigen::Ref<vec_value_t> x
)
{
    for (int i = 0; i < order.size(); ++i) {
        x[order[i]] = x_sorted[i];
    }
} 

ADELIE_CORE_GLM_COX_TP
auto
ADELIE_CORE_GLM_COX::init_packs(
    const std::string& tie_method_str
)
{
    std::vector<pack_t> packs;
    packs.reserve(n_stratas);
    for (size_t i = 0; i < n_stratas; ++i) {
        const auto bi = strata_outer[i];
        const auto si = strata_outer[i+1] - bi;
        packs.emplace_back(
            start_sto.segment(bi, si),
            stop_sto.segment(bi, si),
            status_sto.segment(bi, si),
            weights_sto.segment(bi, si),
            tie_method_str
        );
    }
    return packs;
}

ADELIE_CORE_GLM_COX_TP
ADELIE_CORE_GLM_COX::GlmCox(
    const Eigen::Ref<const vec_value_t>& start,
    const Eigen::Ref<const vec_value_t>& stop,
    const Eigen::Ref<const vec_value_t>& status,
    const Eigen::Ref<const vec_index_t>& strata,
    const Eigen::Ref<const vec_value_t>& weights,
    const std::string& tie_method_str
):
    base_t("cox", status, weights),
    n_stratas(strata.maxCoeff() + 1),
    strata_outer(init_strata_outer(strata, n_stratas)),
    strata_order(init_strata_order(strata)),
    start_sto(init_in_order(start, strata_order)),
    stop_sto(init_in_order(stop, strata_order)),
    status_sto(init_in_order(status, strata_order)),
    weights_sto(init_in_order(weights, strata_order)),
    packs(init_packs(tie_method_str)),
    buffer(3 * status.size())
{
    const auto n = status.size();
    if (start.size() != n) {
        throw util::adelie_core_error("start must be (n,) where status is (n,).");
    }
    if (stop.size() != n) {
        throw util::adelie_core_error("stop must be (n,) where status is (n,).");
    }
    if (strata.size() != n) {
        throw util::adelie_core_error("strata must be (n,) where status is (n,).");
    }
}

ADELIE_CORE_GLM_COX_TP
void
ADELIE_CORE_GLM_COX::gradient(
    const Eigen::Ref<const vec_value_t>& eta,
    Eigen::Ref<vec_value_t> grad
) 
{
    base_t::check_gradient(eta, grad);
    const auto n = eta.size();
    auto eta_sto = buffer.segment(0, n);
    auto grad_sto = buffer.segment(n, n);

    init_in_order(eta, strata_order, eta_sto);

    for (size_t i = 0; i < packs.size(); ++i) {
        auto& pack = packs[i];
        const auto bi = strata_outer[i];
        const auto si = strata_outer[i+1] - bi;
        pack.gradient(
            eta_sto.segment(bi, si),
            grad_sto.segment(bi, si)
        );
    }

    init_from_order(grad_sto, strata_order, grad);
}

ADELIE_CORE_GLM_COX_TP
void
ADELIE_CORE_GLM_COX::hessian(
    const Eigen::Ref<const vec_value_t>& eta,
    const Eigen::Ref<const vec_value_t>& grad,
    Eigen::Ref<vec_value_t> hess
) 
{
    base_t::check_hessian(eta, grad, hess);
    const auto n = eta.size();
    auto eta_sto = buffer.segment(0, n);
    auto grad_sto = buffer.segment(n, n);
    auto hess_sto = buffer.segment(2*n, n);

    init_in_order(eta, strata_order, eta_sto);
    init_in_order(grad, strata_order, grad_sto);
    init_in_order(hess, strata_order, hess_sto);

    for (size_t i = 0; i < packs.size(); ++i) {
        auto& pack = packs[i];
        const auto bi = strata_outer[i];
        const auto si = strata_outer[i+1] - bi;
        pack.hessian(
            eta_sto.segment(bi, si),
            grad_sto.segment(bi, si),
            hess_sto.segment(bi, si)
        );
    }

    init_from_order(hess_sto, strata_order, hess);
}

ADELIE_CORE_GLM_COX_TP
typename ADELIE_CORE_GLM_COX::value_t
ADELIE_CORE_GLM_COX::loss(
    const Eigen::Ref<const vec_value_t>& eta
) 
{
    base_t::check_loss(eta);
    const auto n = eta.size();
    auto eta_sto = buffer.segment(0, n);

    init_in_order(eta, strata_order, eta_sto);

    value_t sum = 0;
    for (size_t i = 0; i < packs.size(); ++i) {
        auto& pack = packs[i];
        const auto bi = strata_outer[i];
        const auto si = strata_outer[i+1] - bi;
        sum += pack.loss(
            eta_sto.segment(bi, si)
        );
    }
    return sum;
}

ADELIE_CORE_GLM_COX_TP
typename ADELIE_CORE_GLM_COX::value_t
ADELIE_CORE_GLM_COX::loss_full() 
{
    value_t sum = 0;
    for (size_t i = 0; i < packs.size(); ++i) {
        auto& pack = packs[i];
        sum += pack.loss_full();
    }
    return sum;
}

ADELIE_CORE_GLM_COX_TP
void
ADELIE_CORE_GLM_COX::inv_link(
    const Eigen::Ref<const vec_value_t>& eta,
    Eigen::Ref<vec_value_t> out 
)
{
    out = eta.exp();
}

} // namespace glm
} // namespace adelie_core