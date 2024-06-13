#pragma once
#include <cmath>
#include <adelie_core/util/types.hpp>

namespace adelie_core {
namespace optimization {

template <class ValueType>
struct QCPSood
{
    using value_t = ValueType;
    using vec_value_t = util::rowvec_type<value_t>;
    using rowarr_value_t = util::rowarr_type<value_t>;

    value_t tol;
    size_t n_threads;

private:
    auto _solve(
        const Eigen::Ref<const vec_value_t>& a,
        const Eigen::Ref<const vec_value_t>& b,
        value_t c,
        const Eigen::Ref<const vec_value_t>& omit,
        value_t lower,
        value_t upper
    )
    {
        const auto p = a.size();

        if (p <= 0) return 0.0;

        vec_value_t buff(p);

        while ((upper - lower) > tol * upper) {
            const auto t = 0.5 * (upper + lower);
            buff = a - t * b;
            std::sort(buff.data(), buff.data() + p, std::greater<value_t>());
            for (int i = 1; i < p; ++i) buff[i] += buff[i-1];
            buff /= vec_value_t::LinSpaced(p, 1, p).sqrt();  
            value_t tc = t * c;
            for (int i = 0; i < p; ++i) {
                if (buff[i] >= tc) {
                    
                }
            }

            a^T x  >= t (b^T x + c)
            x^T (a - tb) >= tc 
        }
    }

public:
    explicit QCPSood(
        value_t tol,
        size_t n_threads
    ):
        tol(tol),
        n_threads(n_threads)
    {
        if (tol <= 0) {
            throw util::adelie_core_error("tol must be > 0.");
        }
        if (n_threads < 1) {
            throw util::adelie_core_error("n_threads must be >= 1.");
        }
    }

    void solve(
        const Eigen::Ref<const rowarr_value_t>& a,
        const Eigen::Ref<const rowarr_value_t>& b,
        const Eigen::Ref<const vec_value_t>& c,
        const Eigen::Ref<const rowarr_value_t>& omit,
        const Eigen::Ref<const vec_value_t>& lower,
        const Eigen::Ref<const vec_value_t>& upper,
        Eigen::Ref<vec_value_t> f
    )
    {
        const auto n = a.rows();

        #pragma omp parallel for schedule(static) num_threads(n_threads) if(n_threads > 1)
        for (int i = 0; i < n; ++i) {
            f[i] = _solve(
                a.row(i), 
                b.row(i), 
                c[i], 
                omit.row(i),
                lower[i],
                upper[i]
            );
        }
    }
};

} // namespace optimization
} // namespace adelie_core