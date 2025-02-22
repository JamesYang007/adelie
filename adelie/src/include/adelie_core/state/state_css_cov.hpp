#pragma once
#include <string>
#include <unordered_set>
#include <adelie_core/util/functional.hpp>
#include <adelie_core/util/types.hpp>

#ifndef ADELIE_CORE_STATE_CSS_COV_TP
#define ADELIE_CORE_STATE_CSS_COV_TP \
    template <\
        class MatrixType,\
        class ValueType,\
        class IndexType\
    >
#endif
#ifndef ADELIE_CORE_STATE_CSS_COV
#define ADELIE_CORE_STATE_CSS_COV \
    StateCSSCov<\
        MatrixType,\
        ValueType,\
        IndexType\
    >
#endif

namespace adelie_core {
namespace state {

template <
    class MatrixType,
    class ValueType=typename std::decay_t<MatrixType>::Scalar,
    class IndexType=Eigen::Index
>
class StateCSSCov
{
public:
    using matrix_t = MatrixType;
    using value_t = ValueType;
    using index_t = IndexType;
    using vec_value_t = util::rowvec_type<value_t>;
    using vec_index_t = util::rowvec_type<index_t>;
    using colmat_value_t = util::colmat_type<value_t>;

    const Eigen::Map<const matrix_t> S;

    /* static states */
    const size_t subset_size;
    const util::css_method_type method;
    const util::css_loss_type loss;

    /* configurations */
    const size_t max_iters;
    const size_t n_threads;

    /* dynamic states */
    std::unordered_set<index_t> subset_set;
    std::vector<index_t> subset;
    colmat_value_t S_resid;
    colmat_value_t L_T;

    static_assert(!matrix_t::IsRowMajor, "Matrix must be column-major!");

    double benchmark_init;
    std::vector<double> benchmark_L_U;
    std::vector<double> benchmark_S_resid;
    std::vector<double> benchmark_scores;
    std::vector<double> benchmark_L_T;
    std::vector<double> benchmark_resid_fwd;

private:
    void initialize();

public:
    virtual ~StateCSSCov() {};

    explicit StateCSSCov(
        const Eigen::Ref<const matrix_t>& S,
        size_t subset_size,
        const Eigen::Ref<const vec_index_t>& init_subset,
        const std::string& method,
        const std::string& loss,
        size_t max_iters,
        size_t n_threads
    ):
        S(S.data(), S.rows(), S.cols()),
        subset_size(subset_size),
        method(util::convert_css_method(method)),
        loss(util::convert_css_loss(loss)),
        max_iters(max_iters),
        n_threads(n_threads),
        subset_set(init_subset.data(), init_subset.data() + init_subset.size()),
        subset(init_subset.data(), init_subset.data() + init_subset.size())
    {
        initialize();
    }

    void solve(
        std::function<void()> check_user_interrupt =util::no_op()
    );
};

} // namespace state
} // namespace adelie_core