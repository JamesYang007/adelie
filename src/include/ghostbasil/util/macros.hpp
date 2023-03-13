#pragma once

 #ifndef GHOSTBASIL_STRONG_INLINE
 #if defined(_MSC_VER)
 #define GHOSTBASIL_STRONG_INLINE __forceinline
 #elif defined(__GNUC__) || defined(__clang__)
 #define GHOSTBASIL_STRONG_INLINE __attribute__((always_inline)) inline
 #else
 #define GHOSTBASIL_STRONG_INLINE inline
 #endif
 #endif

 #ifndef PRINT
 #define PRINT(t)                                                         \
     (std::cerr << std::setprecision(18) << __LINE__ << ": " << #t << '\n' \
                << t << "\n"                                              \
                << std::endl)
 #endif
