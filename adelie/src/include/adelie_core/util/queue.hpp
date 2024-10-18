#pragma once
#include <queue>

namespace adelie_core {
namespace util {

template<
    class T,
    class Container = std::vector<T>,
    class Compare = std::less<typename Container::value_type> 
>
class priority_queue : 
    public std::priority_queue<T, Container, Compare>
{
    using base_t = std::priority_queue<T, Container, Compare>;
public:
    using base_t::base_t;
    using base_t::c;
};

} // namespace util
} // namespace adelie_core
