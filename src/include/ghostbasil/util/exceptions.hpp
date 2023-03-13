#pragma once 
#include <stdexcept>
#include <string>

namespace ghostbasil {
namespace util {

class ghostbasil_error: public std::exception {};

class propagator_error: public ghostbasil_error 
{
    std::string msg_;

public:
    propagator_error() =default;
    propagator_error(const char* msg)
        : msg_(msg)
    {}
     
    const char* what() const noexcept override {
        return msg_.data();
    }
};

class max_cds_error : public ghostbasil_error
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

class max_basil_strong_set : public ghostbasil_error
{
    std::string msg_;
public:
    max_basil_strong_set(): 
        msg_{"Basil maximum strong set size reached."}
    {}

    const char* what() const noexcept override {
        return msg_.data();
    }
};

class basil_finished_early_error : public ghostbasil_error
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

class group_lasso_max_newton_iters : public ghostbasil_error
{
    std::string msg_;
public:
    group_lasso_max_newton_iters():
        msg_{"Max number of Newton iterations reached."}
    {}
    
    const char* what() const noexcept override {
        return msg_.data(); 
    }
};

} // namespace util
} // namespace ghostbasil
