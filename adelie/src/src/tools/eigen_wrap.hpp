#pragma once
// Ignore all warnings for pybind + Eigen
#if defined(_MSC_VER)
#pragma warning( push, 0 )
#elif defined(__GNUC__) 
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall" 
#elif defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall" 
#pragma GCC diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#endif
#include <Eigen/Core>
#include <Eigen/SparseCore>
#if defined(_MSC_VER)
#pragma warning( pop )
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(__clang__)
#pragma GCC diagnostic pop
#pragma GCC diagnostic pop
#endif