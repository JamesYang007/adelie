#include "py_decl.hpp"
#include <algorithm>
#include <atomic>
#include <random>
#include <vector>
#include <state/state.hpp>
#include <adelie_core/util/counting_iterator.hpp>
#include <adelie_core/util/macros.hpp>
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace py = pybind11;
namespace ad = adelie_core;
using namespace pybind11::literals; // to bring in the `_a` literal

template <class SType>
py::dict css_cov_model_selection_fit_k(
    const SType& S,
    size_t k,
    typename SType::Scalar S_logdet,
    typename SType::Scalar cutoff,
    size_t n_inits,
    size_t n_threads,
    size_t seed
)
{
    using matrix_t = std::decay_t<SType>;
    using state_t = ad::state::StateCSSCov<matrix_t>;
    using index_t = typename state_t::index_t;
    using value_t = typename state_t::value_t;
    using vec_index_t = typename state_t::vec_index_t;
    using vec_value_t = typename state_t::vec_value_t;

    constexpr auto inf = std::numeric_limits<value_t>::infinity();

    const auto p = S.cols();

    if (k <= 0 || k >= p-1) {
        throw ad::util::adelie_core_solver_error(
            "k must be in [1, p-1)."
        );
    }

    if (n_threads < 1) {
        throw ad::util::adelie_core_solver_error(
            "n_threads must be >= 1."
        );
    }

    const auto check_user_interrupt = [&]() {
        if (PyErr_CheckSignals() != 0) {
            throw py::error_already_set();
        }
    };

    std::vector<state_t> states;
    std::vector<value_t> best_Ts;
    std::vector<std::vector<index_t>> best_subsets;
    std::vector<std::mt19937> gens;

    // initialize thread data
    for (size_t i = 0; i < n_threads; ++i) {
        states.emplace_back(
            S, 
            k, 
            vec_index_t::LinSpaced(k, 0, k-1),
            "swapping",
            "subset_factor",
            100000,
            1
        );
        best_Ts.emplace_back(inf);
        best_subsets.emplace_back(0);
        gens.emplace_back(((seed+i+1) * 7 * n_inits) % 10007);
    }

    std::atomic_bool early_exit = false; 
    const auto routine = [&](auto i) {
        if (early_exit.load(std::memory_order_relaxed)) return;
        #if defined(_OPENMP)
        const auto thr_idx = omp_get_thread_num();
        #else
        const auto thr_idx = 0;
        #endif
        auto& state = states[thr_idx];
        auto& subset = state.subset;
        auto& subset_set = state.subset_set;
        const auto& S_resid = state.S_resid;
        const auto& L_T = state.L_T;
        auto& gen = gens[thr_idx];
        auto& best_T = best_Ts[thr_idx];
        auto& best_subset = best_subsets[thr_idx];

        // construct the next random subset
        subset.resize(k);
        std::sample(
            ad::util::counting_iterator<index_t>(0),
            ad::util::counting_iterator<index_t>(p),
            subset.begin(),
            k,
            gen
        );
        subset_set.clear();
        subset_set.insert(subset.begin(), subset.end());

        value_t T = -inf;

        // solve CSS with swapping and subset factor loss
        try {
            state.solve(check_user_interrupt);

            // track the best T stat and rejection decision
            T = 2 * L_T.diagonal().array().log().sum() - S_logdet;
            for (int i = 0; i < p; ++i) {
                if (subset_set.find(i) != subset_set.end()) continue;
                const auto S_resid_ii = S_resid(i, i);
                if (S_resid_ii <= 0) {
                    T = -inf;
                    break;
                } 
                T += std::log(S_resid_ii);
            }
        } catch (...) {}

        // check whether to reject the null
        const bool reject = T > cutoff;

        if (T < best_T) {
            best_T = T;
            std::swap(best_subset, subset);
        } 

        if (!reject) early_exit = true;
    };

    if (n_threads <= 1) {
        for (int i = 0; i < static_cast<int>(n_inits); ++i) routine(i);
    } else {
        #if defined(_OPENMP)
        #pragma omp parallel for schedule(static) num_threads(n_threads)
        #endif
        for (int i = 0; i < static_cast<int>(n_inits); ++i) routine(i);
    }

    // find best index
    index_t i_star; 
    Eigen::Map<const vec_value_t>(best_Ts.data(), best_Ts.size()).minCoeff(&i_star);
    const auto& best_subset = best_subsets[i_star];

    return py::dict(
        "T"_a=best_Ts[i_star],
        "subset"_a=vec_index_t(Eigen::Map<const vec_index_t>(best_subset.data(), best_subset.size()))
    );
}

void register_sklearn(py::module_& m)
{
    m.def("css_cov_model_selection_fit_k_32", &css_cov_model_selection_fit_k<dense_type<float, Eigen::ColMajor>>);
    m.def("css_cov_model_selection_fit_k_64", &css_cov_model_selection_fit_k<dense_type<double, Eigen::ColMajor>>);
}