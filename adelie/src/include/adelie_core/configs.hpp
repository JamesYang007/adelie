#pragma once
#include <string>

namespace adelie_core {

struct Configs
{
    static constexpr const char* pb_symbol_def = "\033[1;32m\u2588\033[0m";
    static constexpr const double hessian_min_def = 1e-24;
    static constexpr const double dbeta_tol_def = 1e-12;
    static constexpr const size_t min_bytes_def = 1 << 17;

    inline static std::string pb_symbol = pb_symbol_def;
    inline static double hessian_min = hessian_min_def;
    inline static double dbeta_tol = dbeta_tol_def;
    inline static size_t min_bytes = min_bytes_def;
};

} // namespace adelie_core