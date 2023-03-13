#pragma once

namespace ghostbasil {
namespace util {

/*
 * Functor that represents a no-op.
 * It takes in any number of arguments of any type and does nothing.
 */
struct no_op
{
    template <class... Args>
    void operator()(Args&&... args) {}
};

} // namespace util
} // namespace ghostbasil
