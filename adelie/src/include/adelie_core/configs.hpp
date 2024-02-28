#pragma once
#include <string>

namespace adelie_core {

struct Configs
{
    static constexpr const char* pb_symbol_def = "\033[1;32m\u2588\033[0m";

    inline static std::string pb_symbol = pb_symbol_def;
};

} // namespace adelie_core