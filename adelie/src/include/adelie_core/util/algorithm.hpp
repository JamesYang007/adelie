#pragma once
#include <numeric>
#include <adelie_core/util/queue.hpp>

namespace adelie_core {
namespace util {

template <class VecType, class SkipType, class Iter>
inline size_t k_imax(
        const VecType& v,
        const SkipType& skip,
        size_t k,
        Iter out_begin)
{
    using idx_t = typename Iter::value_type;
    
    if (k <= 0) return 0;

    auto comp = [&](auto i, auto j) {
        return v[i] > v[j];
    };
    using comp_t = std::decay_t<decltype(comp)>;

    size_t n_added = 0;

    // PQ on -v values (k smallest values of -v)
    priority_queue<idx_t, std::vector<idx_t>, comp_t> pq(comp);

    size_t i = 0;
    for (; n_added < k && i < v.size(); ++i) {
        if (skip(i)) continue;
        pq.push(i);
        ++n_added;
    }

    if (n_added >= k) {
        for (; i < v.size(); ++i) {
            if (skip(i) || v[pq.top()] >= v[i]) continue;
            pq.pop();
            pq.push(i);
        }
    }

    std::copy(pq.c.begin(), pq.c.end(), out_begin);

    return n_added;
}

} // namespace util
} // namespace adelie_core
