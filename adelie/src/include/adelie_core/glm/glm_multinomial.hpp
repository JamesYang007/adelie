#pragma once
#include <adelie_core/glm/glm_base.hpp>

namespace adelie_core {
namespace glm {

template <class ValueType>
class GlmMultinomial: public GlmBase<ValueType>
{
public:
    using base_t = GlmBase<ValueType>;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;

private:
    size_t _K;
    vec_value_t _buff;

public:
    explicit GlmMultinomial(
        size_t K
    ):
        base_t("multinomial", true),
        _K(K)
    {}

    void gradient(
        const Eigen::Ref<const vec_value_t>& eta,
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> mu
    ) override
    {
        Eigen::Map<const util::rowarr_type<value_t>> W(
            weights.data(), weights.size() / _K, _K
        );
        Eigen::Map<const util::rowarr_type<value_t>> E(
            eta.data(), W.rows(), W.cols()
        );
        Eigen::Map<util::rowarr_type<value_t>> M(
            mu.data(), W.rows(), W.cols()
        );
        _buff.resize(W.rows());

        M = E.exp();
        _buff = W.col(0) / (1 + M.rowwise().sum());
        M.colwise() *= _buff.matrix().transpose().array();
    }

    void hessian(
        const Eigen::Ref<const vec_value_t>& mu,
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> var
    ) override
    {
        Eigen::Map<const util::rowarr_type<value_t>> W(
            weights.data(), weights.size() / _K, _K
        );
        Eigen::Map<const util::rowarr_type<value_t>> M(
            mu.data(), W.rows(), W.cols()
        );
        Eigen::Map<util::rowarr_type<value_t>> V(
            var.data(), W.rows(), W.cols()
        );

        V = (W - M) * M / (W + (W <= 0).template cast<value_t>());
    }

    value_t deviance(
        const Eigen::Ref<const vec_value_t>& y,
        const Eigen::Ref<const vec_value_t>& eta,
        const Eigen::Ref<const vec_value_t>& weights
    ) override
    {
        Eigen::Map<const util::rowarr_type<value_t>> W(
            weights.data(), weights.size() / _K, _K
        );
        Eigen::Map<const util::rowarr_type<value_t>> E(
            eta.data(), W.rows(), W.cols()
        );
        return (
            - (weights * y * eta).sum()
            + ((1 + E.exp().rowwise().sum()).log() * W.col(0)).sum()
        );
    }

    value_t deviance_full(
        const Eigen::Ref<const vec_value_t>&,
        const Eigen::Ref<const vec_value_t>& 
    ) override
    {
        return 0;
    }
};

} // namespace glm
} // namespace adelie_core