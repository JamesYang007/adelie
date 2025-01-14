#pragma once
#include <adelie_core/util/exceptions.hpp>
#include <adelie_core/util/types.hpp>
#include <adelie_core/optimization/newton.hpp>

namespace adelie_core {
namespace bcd {
namespace sgl {

template <class T>
auto cbrt_complex(const std::complex<T>& z) 
{
    const auto r = std::abs(z);
    const auto theta = std::arg(z);
    const auto r_cbrt = std::cbrt(r);
    const auto theta_cbrt = theta / 3.0;
    return std::polar(r_cbrt, theta_cbrt);
}

// https://en.wikipedia.org/wiki/Quartic_equation#The_general_case
template <class ValueType>
inline
util::rowvec_type<std::complex<ValueType>, 4> quartic_roots(
    ValueType A,
    ValueType B,
    ValueType C,
    ValueType D,
    ValueType E
)
{
    using value_t = ValueType;
    using complex_t = std::complex<value_t>;
    using roots_t = util::rowvec_type<complex_t, 4>;

    if (A == 0) {
        throw util::adelie_core_solver_error(
            "TODO: cubic real roots must be implemented!"
        );
    }

    roots_t roots;

    // convert to depressed quartic
    // eliminate x^3 term by transforming x = u - B/4
    // solving u^4 + a u^2 + b u + c = 0
    B /= A;
    C /= A;
    D /= A;
    E /= A;
    const auto B_sq = B * B;
    const auto a = (-3./8) * B_sq + C;
    const auto b = B * (B_sq / 8. - 0.5 * C) + D;
    const auto c = B_sq * ((-3./256) * B_sq + C / 16.) - 0.25 * B * D + E;

    // special case: biquadratic
    if (b == 0) {
        const auto discr = a * a - 4 * c;
        if (discr >= 0) {
            const auto sqrt_discr = std::sqrt(discr);
            const auto zp = 0.5 * (-a + sqrt_discr);
            const auto zm = 0.5 * (-a - sqrt_discr);
            const auto sqrt_zp = std::sqrt(std::abs(zp));
            const auto sqrt_zm = std::sqrt(std::abs(zm));
            if (zp >= 0) {
                roots[0] = complex_t(sqrt_zp, 0);
                roots[1] = complex_t(-sqrt_zp, 0);
            } else {
                roots[0] = complex_t(0, sqrt_zp);
                roots[1] = complex_t(0, -sqrt_zp);
            }
            if (zm >= 0) {
                roots[2] = complex_t(sqrt_zm, 0);
                roots[3] = complex_t(-sqrt_zm, 0);
            } else {
                roots[2] = complex_t(0, sqrt_zm);
                roots[3] = complex_t(0, -sqrt_zm);
            }
        }

    // general case: b != 0
    } else {
        const auto a_sq = a * a;
        const auto p = -a_sq / 12. - c;
        const auto q = -a * (a_sq / 108. - c / 3.) - b * b / 8.;
        const auto s = -0.5 * q;
        const auto t_sq = s * s + p * p * p / 27.;
        const auto t_abs = std::sqrt(std::abs(t_sq));
        const auto t = (t_sq >= 0) ? complex_t(t_abs, 0) : complex_t(0, t_abs);
        // s * Re(t) >= 0 if and only if |s+t| >= |s-t| (assuming t >= 0)
        // choose the w that has the largest magnitude
        const auto w_cube = s + (2. * (s * std::real(t) >= 0) - 1.) * t;
        const auto w = cbrt_complex(w_cube);
        const auto y = a / 6. + w - p / (3. * w); // TODO: w != 0 check?
        const auto z = std::sqrt(2. * y - a);
        const auto rp = -2. * y - a + 2 * b / z;
        const auto rm = -2. * y - a - 2 * b / z;
        const auto sqrt_rp = std::sqrt(rp);
        roots[0] = 0.5 * (-z + sqrt_rp);
        roots[1] = 0.5 * (-z - sqrt_rp);
        const auto sqrt_rm = std::sqrt(rm);
        roots[2] = 0.5 * (z + sqrt_rm);
        roots[3] = 0.5 * (z - sqrt_rm);
    }

    roots -= 0.25 * B;
    return roots;
}

/**
 * Solves the positive root of the SGL secular equation at f(x) = y:
 * 
 *      f(x) := (m + b / sqrt(x^2 + a)) x
 * 
 * User must verify that m, a, b >= 0.
 */
template <class ValueType>
auto root_secular(
    ValueType y,
    ValueType m,
    ValueType a,
    ValueType b,
    ValueType tol,
    size_t max_iters
)
{
    using value_t = ValueType;

    const auto h_cand = (y - b) / m;

    if (a <= 0 || b <= 0) {
        if (m <= 0) {
            throw util::adelie_core_solver_error(
                "root_secular: m = a = 0 is invalid!"
            );
        }
        return h_cand;
    }

    const auto initial_f = [&](){ return std::make_pair(h_cand, 0); };

    const auto step_f = [&](auto h) {
        const auto u1 = h * h + a;
        const auto u2 = b / std::sqrt(u1);
        const auto fh = (m + u2) * h - y;
        const auto dfh = m + u2 * a / u1;
        return std::make_pair(fh, dfh);
    };

    const auto project_f = [&](auto h) {
        return std::max<value_t>(h, 0.0);
    };

    const auto root_find_state = optimization::newton_root_find(
        initial_f,
        step_f,
        project_f,
        tol, 
        max_iters
    );

    return std::get<0>(root_find_state);
}

} // namespace sgl
} // namespace bcd
} // namespace adelie_core