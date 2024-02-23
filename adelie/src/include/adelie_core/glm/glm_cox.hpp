#pragma once
#include <adelie_core/glm/glm_base.hpp>
#include <adelie_core/util/types.hpp>

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

template <class ValueType>
class GlmCox: public GlmBase<ValueType>
{
public:
    using base_t = GlmBase<ValueType>;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using typename base_t::map_cvec_value_t;
    using index_t = int;
    using vec_index_t = util::rowvec_type<index_t>;

    const util::tie_method_type tie_method;

    /* original order quantities */
    const map_cvec_value_t start;
    const map_cvec_value_t stop;
    const std::decay_t<decltype(base_t::y)>& status = base_t::y;
    using base_t::weights;

    /* start order quantities (sorted by start time) */
    const vec_index_t start_order;
    const vec_value_t start_so;

    /* stop order quantities (sorted by stop time) */
    const vec_index_t stop_order;
    const vec_value_t stop_to;
    const vec_value_t status_to;
    const vec_value_t weights_to;
    const vec_value_t weights_size_to;
    const vec_value_t weights_mean_to;
    const vec_value_t scale_to;

    /* buffers */
    vec_value_t buffer;

private:
    static auto init_order(
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

    static auto init_in_order(
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

    static auto init_weights_size_to(
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

    static auto init_weights_mean_to(
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

    static auto init_scale_to(
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

public:
    explicit GlmCox(
        const Eigen::Ref<const vec_value_t>& start,
        const Eigen::Ref<const vec_value_t>& stop,
        const Eigen::Ref<const vec_value_t>& status,
        const Eigen::Ref<const vec_value_t>& weights,
        const std::string& tie_method_str
    ):
        base_t("cox", status, weights),
        tie_method(util::convert_tie_method(tie_method_str)),
        start(start.data(), start.size()),
        stop(stop.data(), stop.size()),
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
    {
        const auto n = start.size();
        if (stop.size() != n) {
            throw std::runtime_error("stop vector must be same length as start.");
        }
        if (status.size() != n) {
            throw std::runtime_error("status vector must be same length as start.");
        }
        if (weights.size() != n) {
            throw std::runtime_error("weights vector must be same length as start.");
        }
    }

    void gradient(
        const Eigen::Ref<const vec_value_t>& eta,
        Eigen::Ref<vec_value_t> grad
    ) override
    {
        base_t::check_gradient(eta, grad);
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

    void hessian(
        const Eigen::Ref<const vec_value_t>& eta,
        const Eigen::Ref<const vec_value_t>& grad,
        Eigen::Ref<vec_value_t> hess
    ) override
    {
        base_t::check_hessian(eta, grad, hess);
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

    value_t loss(
        const Eigen::Ref<const vec_value_t>& eta
    ) override
    {
        base_t::check_loss(eta);
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

    value_t loss_full() override
    {
        const constexpr auto most_neg = -std::numeric_limits<value_t>::max();
        return (
            weights_mean_to * status_to * 
            (weights_size_to * weights_mean_to * (1 - scale_to)).log().max(most_neg)
        ).sum();
    }
};

} // namespace glm
} // namespace adelie_core