#pragma once
#include <adelie_core/glm/glm_base.hpp>

#ifndef ADELIE_CORE_GLM_COX_PACK_TP
#define ADELIE_CORE_GLM_COX_PACK_TP \
    template <class ValueType, class IndexType>
#endif
#ifndef ADELIE_CORE_GLM_COX_PACK
#define ADELIE_CORE_GLM_COX_PACK \
    GlmCoxPack<ValueType, IndexType>
#endif

#ifndef ADELIE_CORE_GLM_COX_TP
#define ADELIE_CORE_GLM_COX_TP \
    template <class ValueType, class IndexType>
#endif
#ifndef ADELIE_CORE_GLM_COX
#define ADELIE_CORE_GLM_COX \
    GlmCox<ValueType, IndexType>
#endif

namespace adelie_core {
namespace glm {

template <class ValueType, class IndexType=Eigen::Index>
class GlmCoxPack
{
public:
    using index_t = IndexType;
    using value_t = ValueType;
    using vec_index_t = util::rowvec_type<index_t>;
    using vec_value_t = util::rowvec_type<value_t>;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;

    const util::tie_method_type tie_method;

    /* original order quantities */
    const map_cvec_value_t start;
    const map_cvec_value_t stop;
    const map_cvec_value_t status;
    const map_cvec_value_t weights;

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
    explicit GlmCoxPack(
        const Eigen::Ref<const vec_value_t>& start,
        const Eigen::Ref<const vec_value_t>& stop,
        const Eigen::Ref<const vec_value_t>& status,
        const Eigen::Ref<const vec_value_t>& weights,
        const std::string& tie_method_str
    );

    inline void gradient(
        const Eigen::Ref<const vec_value_t>& eta,
        Eigen::Ref<vec_value_t> grad
    );

    inline void hessian(
        const Eigen::Ref<const vec_value_t>& eta,
        const Eigen::Ref<const vec_value_t>& grad,
        Eigen::Ref<vec_value_t> hess
    );

    inline value_t loss(
        const Eigen::Ref<const vec_value_t>& eta
    );

    inline value_t loss_full();
};

template <class ValueType, class IndexType=Eigen::Index>
class GlmCox: public GlmBase<ValueType>
{
public:
    using base_t = GlmBase<ValueType>;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using typename base_t::map_cvec_value_t;
    using pack_t = GlmCoxPack<value_t, IndexType>;
    using index_t = typename pack_t::index_t;
    using vec_index_t = typename pack_t::vec_index_t;

    /* strata order quantities */
    const size_t n_stratas;
    const vec_index_t strata_outer;
    const vec_index_t strata_order;
    const vec_value_t start_sto;
    const vec_value_t stop_sto;
    const vec_value_t status_sto;
    const vec_value_t weights_sto;

    std::vector<pack_t> packs;

    /* buffers */
    vec_value_t buffer;

private:
    static inline auto init_strata_outer(
        const Eigen::Ref<const vec_index_t>& strata,
        size_t n_stratas
    );

    static inline auto init_strata_order(
        const Eigen::Ref<const vec_index_t>& strata
    );

    static inline void init_in_order(
        const Eigen::Ref<const vec_value_t>& x,
        const Eigen::Ref<const vec_index_t>& order,
        Eigen::Ref<vec_value_t> x_sorted
    );

    static inline auto init_in_order(
        const Eigen::Ref<const vec_value_t>& x,
        const Eigen::Ref<const vec_index_t>& order
    );

    static inline void init_from_order(
        const Eigen::Ref<const vec_value_t>& x_sorted,
        const Eigen::Ref<const vec_index_t>& order,
        Eigen::Ref<vec_value_t> x
    );

    inline auto init_packs(
        const std::string& tie_method_str
    );

public:
    explicit GlmCox(
        const Eigen::Ref<const vec_value_t>& start,
        const Eigen::Ref<const vec_value_t>& stop,
        const Eigen::Ref<const vec_value_t>& status,
        const Eigen::Ref<const vec_index_t>& strata,
        const Eigen::Ref<const vec_value_t>& weights,
        const std::string& tie_method_str
    );

    ADELIE_CORE_GLM_PURE_OVERRIDE_DECL
};

} // namespace glm
} // namespace adelie_core