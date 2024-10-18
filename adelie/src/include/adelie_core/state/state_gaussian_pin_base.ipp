#pragma once
#include <adelie_core/state/state_gaussian_pin_base.hpp>

namespace adelie_core {
namespace state {

ADELIE_CORE_STATE_GAUSSIAN_PIN_BASE_TP
void
ADELIE_CORE_STATE_GAUSSIAN_PIN_BASE::initialize()
{
    active_begins.reserve(screen_set.size());
    int active_begin = 0;
    for (size_t i = 0; i < active_set_size; ++i) {
        const auto ia = active_set[i];
        const auto curr_size = group_sizes[screen_set[ia]];
        active_begins.push_back(active_begin);
        active_begin += curr_size;
    }

    active_order.resize(active_set_size);
    std::iota(active_order.begin(), active_order.end(), 0);
    std::sort(
        active_order.begin(),
        active_order.end(),
        [&](auto i, auto j) { 
            return groups[screen_set[active_set[i]]] < groups[screen_set[active_set[j]]]; 
        }
    );

    betas.reserve(lmda_path.size());
    intercepts.reserve(lmda_path.size());
    rsqs.reserve(lmda_path.size());
    lmdas.reserve(lmda_path.size());
    benchmark_screen.reserve(1000);
    benchmark_active.reserve(1000);
}

} // namespace state
} // namespace adelie_core