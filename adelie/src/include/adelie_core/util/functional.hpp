#pragma once

namespace adelie_core {
namespace util {

/*
 * Functor that represents a no-op.
 * It takes in any number of arguments of any type and does nothing.
 */
struct no_op
{
    // This hides a lot of bugs with multiple functor passed in...
    //template <class... Args>
    //void operator()(Args&&... args) {}
    template <class T>
    void operator()(T) {}

    void operator()() {}
};

} // namespace util
} // namespace adelie_core
