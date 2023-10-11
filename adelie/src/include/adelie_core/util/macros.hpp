#pragma once

#ifndef ADELIE_CORE_STRONG_INLINE
#if defined(_MSC_VER)
#define ADELIE_CORE_STRONG_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define ADELIE_CORE_STRONG_INLINE __attribute__((always_inline)) inline
#else
#define ADELIE_CORE_STRONG_INLINE inline
#endif
#endif

#ifndef PRINT
#include <iostream>
#include <iomanip>
#define PRINT(t)                                                         \
    (std::cerr << std::setprecision(18) << __LINE__ << ": " << #t << '\n' \
            << t << "\n"                                              \
            << std::endl)
#endif
