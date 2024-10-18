#pragma once
#include <adelie_core/glm/glm_base.hpp>

#ifndef ADELIE_CORE_GLM_COX_TP
#define ADELIE_CORE_GLM_COX_TP \
    template <class ValueType>
#endif
#ifndef ADELIE_CORE_GLM_COX
#define ADELIE_CORE_GLM_COX \
    GlmCox<ValueType>
#endif

namespace adelie_core {
namespace glm {

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
    static inline auto init_order(
        const Eigen::Ref<const vec_value_t>& x
    );

    static inline auto init_in_order(
        const Eigen::Ref<const vec_value_t>& x,
        const Eigen::Ref<const vec_index_t>& order
    );

    static inline auto init_weights_size_to(
        const Eigen::Ref<const vec_value_t>& stop_to,
        const Eigen::Ref<const vec_value_t>& status_to,
        const Eigen::Ref<const vec_value_t>& weights_to
    );

    static inline auto init_weights_mean_to(
        const Eigen::Ref<const vec_value_t>& stop_to,
        const Eigen::Ref<const vec_value_t>& status_to,
        const Eigen::Ref<const vec_value_t>& weights_to,
        const Eigen::Ref<const vec_value_t>& weights_size_to
    );

    static inline auto init_scale_to(
        const Eigen::Ref<const vec_value_t>& stop_to,
        const Eigen::Ref<const vec_value_t>& status_to,
        const Eigen::Ref<const vec_value_t>& weights_to,
        util::tie_method_type tie_method
    );

public:
    explicit GlmCox(
        const Eigen::Ref<const vec_value_t>& start,
        const Eigen::Ref<const vec_value_t>& stop,
        const Eigen::Ref<const vec_value_t>& status,
        const Eigen::Ref<const vec_value_t>& weights,
        const std::string& tie_method_str
    );

    ADELIE_CORE_GLM_PURE_OVERRIDE_DECL
};

} // namespace glm
} // namespace adelie_core