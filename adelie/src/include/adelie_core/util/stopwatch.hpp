#pragma once
#include <chrono>

namespace adelie_core {
namespace util {
    
class Stopwatch
{
    using sw_clock_t = std::chrono::steady_clock;
    using tpt_t = std::chrono::time_point<sw_clock_t>;
    double& store_;
    double elapsed_;
    tpt_t start_; 

public:
    Stopwatch(double& store)
        : store_(store)
    {
        start();
    }
    
    ~Stopwatch()
    {
        elapsed();
        store_ = elapsed_;
    }
    
    void start()
    {
        start_ = sw_clock_t::now();    
    }
    
    void elapsed()
    {
        const auto end = sw_clock_t::now();
        const auto dur = (end - start_);
        elapsed_ = std::chrono::duration_cast<std::chrono::nanoseconds>(dur).count() * 1e-9;
    }
};

} // namespace util
} // namespace adelie_core