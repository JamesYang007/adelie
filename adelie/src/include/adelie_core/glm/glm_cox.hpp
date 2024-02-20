#pragma once
#include <adelie_core/glm/glm_base.hpp>

namespace adelie_core {
namespace glm {
namespace cox {

/**
 * Computes the partial sums based on the thresholds given by ``t`` and time-points ``s``.
 * 
 *      out[i] = \sum_{k=1}^n 1_{s_k \geq t_i} v_k
 * 
 * @param v     (n,) array of values to partial sum.
 * @param s     (n,) array of time-points in increasing order.
 * @param t     (m,) array of thresholds in increasing order.
 * @param out   (m+1,) array of partial sums where the last element is a padding of 0,
 *              i.e. out[m] == 0.
 */
template <class VType, class SType, class TType, class OutType>
void _partial_sum(
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
        for (; k_begin >= 0 && s[k_begin] >= ti; --k_begin) curr_sum += v[k_begin];
        // save current sum for all positions with time == t_i
        for (; i_end >= 0 && t[i_end] == ti; --i_end) out[i_end] = curr_sum;
        i_begin = i_end;
        if (k_begin < 0) break;
    }

    for (; i_begin >= 0; --i_begin) out[i_begin] = out[i_begin+1];
}

/**
 * Computes the average of w among ties and non-zero values.
 * If w[i] != 0, then
 *      out[i] = \sum_{k=1}^n 1_{t_k = t_i} w_k / \sum_{k=1}^n 1_{t_k = t_i, w_k \neq 0}
 * Otherwise,
 *      out[i] = 0
 * 
 * @param   w   (n,) vector of values to average among ties and non-zero values.
 * @param   t   (n,) vector of thresholds in increasing order.
 * @param   out (n,) vector of outputs.
 */
template <class WType, class TType, class OutType>
void _average_ties(
    const WType& w,
    const TType& t,
    OutType& out
)
{
    using value_t = typename std::decay_t<WType>::Scalar;
    const auto n = w.size();
    int i_begin = 0;
    int i_end = 0;
    while (i_begin < n) {
        const auto ti = t[i_begin];
        value_t sum = 0;
        int size = 0;
        while (i_end < n && t[i_end] == ti) {
            sum += w[i_end];
            size += w[i_end] != 0;
            ++i_end;
        }
        const auto mean = sum / size;
        for (int j = i_begin; j < i_end; ++j) {
            out[j] = mean * (w[j] != 0);
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
    using base_t::y;
    using base_t::weights;

    const map_cvec_value_t start;
    const map_cvec_value_t stop;
    const vec_index_t stop_order;
    const vec_value_t stop_sorted;
    vec_value_t buffer1_n;
    vec_value_t buffer2_n;
    vec_value_t buffer3_n;

private:
    static auto init_stop_order(
        const Eigen::Ref<const vec_value_t>& stop
    )
    {
        vec_index_t stop_order = vec_index_t::LinSpaced(stop.size(), 0, stop.size()-1);
        std::sort(
            stop_order.data(), 
            stop_order.data() + stop_order.size(),
            [&](auto i, auto j) { return stop[i] < stop[j]; }
        );
        return stop_order;
    }

    static auto init_stop_sorted(
        const Eigen::Ref<const vec_value_t>& stop,
        const Eigen::Ref<const vec_index_t>& order
    )
    {
        vec_value_t stop_sorted(stop.size());
        for (int i = 0; i < order.size(); ++i) {
            stop_sorted[i] = stop[order[i]];
        }
        return stop_sorted;
    } 

public:
    explicit GlmCox(
        const Eigen::Ref<const vec_value_t>& start,
        const Eigen::Ref<const vec_value_t>& stop,
        const Eigen::Ref<const vec_value_t>& status,
        const Eigen::Ref<const vec_value_t>& weights
    ):
        base_t("cox", status, weights),
        start(start.data(), start.size()),
        stop(stop.data(), stop.size()),
        stop_order(init_stop_order(stop)),
        stop_sorted(init_stop_sorted(stop, stop_order)),
        buffer1_n(start.size()),
        buffer2_n(start.size()),
        buffer3_n(start.size())
    {
        const auto n = start.size();
        if (stop.size() != n) {
            throw std::runtime_error("stop vector must be same length as start.");
        }
        if (status.size() != n) {
            throw std::runtime_error("status vector must be same length as start.");
        }
        // don't check weights size since it can be lazily assigned
    }

    void gradient(
        const Eigen::Ref<const vec_value_t>& eta,
        Eigen::Ref<vec_value_t> grad
    ) override
    {
        base_t::check_gradient(eta, grad);
    }

    void hessian(
        const Eigen::Ref<const vec_value_t>& eta,
        const Eigen::Ref<const vec_value_t>& grad,
        Eigen::Ref<vec_value_t> hess
    ) override
    {
        base_t::check_hessian(eta, grad, hess);
    }

    value_t loss(
        const Eigen::Ref<const vec_value_t>& eta
    ) override
    {
        base_t::check_loss(eta);
        return 0;
    }

    value_t loss_full() override
    {
        return 0;
    }
};

} // namespace glm
} // namespace adelie_core