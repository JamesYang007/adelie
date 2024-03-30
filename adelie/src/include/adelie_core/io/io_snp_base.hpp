#pragma once
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <adelie_core/util/types.hpp>
#if defined(__linux__) || defined(__APPLE__)
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace adelie_core {
namespace io {

class IOSNPBase
{
public:
    using string_t = std::string;
    using file_unique_ptr_t = std::unique_ptr<
        std::FILE, 
        std::function<void(std::FILE*)>
    >;
    using mmap_unique_ptr_t = std::unique_ptr<
        char,
        std::function<void(char*)>
    >;
    using bool_t = bool;
    using buffer_t = util::rowvec_type<char>;

protected:
    const string_t _filename;
    const util::read_mode_type _read_mode;
    buffer_t _buffer_w;             // only used when _read_mode == _file
    mmap_unique_ptr_t _mmap_ptr;    // only used when _read_mode == _mmap
    Eigen::Map<buffer_t> _buffer;
    bool_t _is_read;

    static void throw_no_read() 
    {
        throw std::runtime_error(
            "File is not read yet. Call read() first."
        );
    }
    
    static auto fopen_safe(
        const char* filename,
        const char* mode
    )
    {
        file_unique_ptr_t file_ptr(
            std::fopen(filename, mode),
            [](std::FILE* fp) { std::fclose(fp); }
        );
        auto fp = file_ptr.get();
        if (!fp) {
            throw std::runtime_error("Cannot open file " + std::string(filename));
        }
        // disable internal buffering
        std::setvbuf(fp, nullptr, _IONBF, 0);
        return file_ptr;
    }

    static bool is_big_endian() 
    {
        union {
            uint32_t i;
            char c[4];
        } _bint = {0x01020304};

        return _bint.c[0] == 1;
    }

    static auto convert_read_mode(
        const std::string& read_mode
    )
    {
        auto read_mode_enum = util::convert_read_mode(read_mode);
        if (read_mode_enum == util::read_mode_type::_auto) {
#if defined(__linux__) || defined(__APPLE__)
            read_mode_enum = util::read_mode_type::_mmap;
#else
            read_mode_enum = util::read_mode_type::_file;
#endif
        }
        return read_mode_enum;
    }

public:
    IOSNPBase(
        const string_t& filename,
        const string_t& read_mode
    ):
        _filename(filename),
        _read_mode(convert_read_mode(read_mode)),
        _mmap_ptr(nullptr, [](char*) {}),
        _buffer(nullptr, 0),
        _is_read(false)
    {}

    bool_t endian() const { 
        if (!_is_read) throw_no_read();
        return reinterpret_cast<const bool_t&>(_buffer[0]); 
    }

    size_t read()
    {
        _is_read = true;

        // compute the total number of bytes
        auto file_ptr = fopen_safe(_filename.c_str(), "rb");
        auto fp = file_ptr.get();
        std::fseek(fp, 0, SEEK_END);
        const size_t total_bytes = std::ftell(fp);
        std::fseek(fp, 0, SEEK_SET);

        // use the optimized mmap routine
        if (_read_mode == util::read_mode_type::_mmap) {
#if defined(__linux__) || defined(__APPLE__)
            int fd = open(_filename.c_str(), O_RDONLY);
            char* addr = static_cast<char*>(
                mmap(
                    nullptr, 
                    total_bytes, 
                    PROT_READ,
                    MAP_PRIVATE,
                    fd,
                    0
                )
            );
            close(fd);
            _mmap_ptr = mmap_unique_ptr_t(
                addr, 
                [=](char* ptr) {
                    if (ptr) {
                        munmap(ptr, total_bytes);
                    }
                }
            );
            new (&_buffer) Eigen::Map<buffer_t>(addr, total_bytes);
#else
            throw std::runtime_error("Only Linux and MacOS support the mmap feature.");
#endif
        // otherwise use the more general routine using file IO
        } else if (_read_mode == util::read_mode_type::_file) {
            _buffer_w.resize(total_bytes);
            const size_t read = std::fread(_buffer_w.data(), sizeof(char), _buffer_w.size(), fp);
            if (read != static_cast<size_t>(_buffer_w.size())) {
                throw std::runtime_error(
                    "Could not read the whole file into buffer."
                );
            }
            new (&_buffer) Eigen::Map<buffer_t>(_buffer_w.data(), _buffer_w.size());

        } else {
            throw std::runtime_error("Unsupported read mode.");
        }

        if (endian() != is_big_endian()) {
            throw std::runtime_error(
                "Endianness is inconsistent! "
                "Regenerate the file on a machine with the same endianness."
            );
        }

        return total_bytes;
    }

};

} // namespace io
} // namespace adelie_core