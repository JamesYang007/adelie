#pragma once 
#include <stdexcept>
#include <string>

namespace adelie_core {
namespace util {

class adelie_core_error: public std::exception 
{
    std::string _msg;

public:
    adelie_core_error(
        const std::string& msg
    ):
        _msg("adelie_core: " + msg)
    {}

    adelie_core_error(
        const std::string& prefix,
        const std::string& msg
    ):
        _msg("adelie_core " + prefix + ": " + msg)
    {}

    const char* what() const noexcept override {
        return _msg.data();
    }
};

class adelie_core_solver_error: public adelie_core_error
{
public:
    adelie_core_solver_error(
        const std::string& msg
    ):
        adelie_core_error("solver", msg)
    {}
};

class max_cds_error : public adelie_core_solver_error
{
public:
    max_cds_error(int lmda_idx)
        : adelie_core_solver_error("max coordinate descents reached at lambda index: " + std::to_string(lmda_idx) + ".")
    {}
};

class max_screen_set_error : public adelie_core_solver_error
{
public:
    max_screen_set_error(): 
        adelie_core_solver_error("maximum screen set size reached.")
    {}
};

} // namespace util
} // namespace adelie_core
