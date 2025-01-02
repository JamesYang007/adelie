#pragma once
#include <adelie_core/bcd/utils.hpp>
#include <adelie_core/optimization/bisect.hpp>
#include <adelie_core/util/types.hpp>

namespace adelie_core {
namespace bcd {
namespace unconstrained {

template <class LType, class VType, class ValueType, class XType>
void brent_solver(
    const LType& L,
    const VType& v,
    ValueType l1,
    ValueType l2,
    ValueType tol,
    size_t max_iters,
    XType& x,
    size_t& iters
)
{
    using value_t = ValueType;
    value_t h = 0;
    const util::rowvec_type<value_t> buffer1 = L + l2;
    const auto a = root_lower_bound(buffer1, v, l1);
    const auto b = std::get<0>(root_upper_bound(buffer1, v, l1, 0.0));
    iters = 0;
    const auto phi = [&](auto h) {
        return root_function(h, buffer1, v, l1);
    };
    optimization::brent(
        phi, tol, tol, max_iters, a, a, b, 
        [](auto, auto, auto, auto) { return std::make_pair(false, 0.0); },
        h, iters
    );
    x = h * v / (buffer1 * h + l1);
}

} // namespace unconstrained
} // namespace bcd
} // namespace adelie_core