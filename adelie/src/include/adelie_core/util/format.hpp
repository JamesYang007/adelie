#pragma once
#include <memory>
#include <stdexcept>
#include <string>
#include <adelie_core/util/exceptions.hpp>

namespace adelie_core {
namespace util {

template<typename ... Args>
std::string format(
    const char* fmt, 
    Args ... args
)
{
    int size_s = std::snprintf( nullptr, 0, fmt, args ... ) + 1; // Extra space for '\0'
    if (size_s <= 0) throw util::adelie_core_error("Error during formatting.");
    auto size = static_cast<size_t>(size_s);
    std::unique_ptr<char[]> buf( new char[ size ] );
    std::snprintf( buf.get(), size, fmt, args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

} // namespace util
} // namespace adelie_core