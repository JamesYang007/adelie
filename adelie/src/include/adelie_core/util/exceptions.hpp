#pragma once 
#include <stdexcept>
#include <string>

namespace adelie_core {
namespace util {

class adelie_core_error: public std::exception {};

class max_cds_error : public adelie_core_error
{
    std::string msg_;

public:
    max_cds_error(int lmda_idx)
        : msg_{"Basil max coordinate descents reached at lambda index: " + std::to_string(lmda_idx) + "."}
    {}

    const char* what() const noexcept override {
        return msg_.data();
    }
};

class max_basil_screen_set : public adelie_core_error
{
    std::string msg_;
public:
    max_basil_screen_set(): 
        msg_{"Basil maximum screen set size reached."}
    {}

    const char* what() const noexcept override {
        return msg_.data();
    }
};

class basil_finished_early_error : public adelie_core_error
{
    std::string msg_;
public:
    basil_finished_early_error(): 
        msg_{"Basil finished early due to minimal change in R^2."}
    {}

    const char* what() const noexcept override {
        return msg_.data();
    }
};

class group_elnet_max_newton_iters : public adelie_core_error
{
    std::string msg_;
public:
    group_elnet_max_newton_iters():
        msg_{"Max number of Newton iterations reached."}
    {}
    
    const char* what() const noexcept override {
        return msg_.data(); 
    }
};

} // namespace util
} // namespace adelie_core
