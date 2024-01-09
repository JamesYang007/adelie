#pragma once
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <adelie_core/util/types.hpp>

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
    using bool_t = bool;
    using buffer_t = util::rowvec_type<char>;

protected:
    const string_t _filename;
    buffer_t _buffer;
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

public:
    IOSNPBase(
        const string_t& filename
    ):
        _filename(filename),
        _buffer(),
        _is_read(false)
    {}

    bool_t endian() const { 
        if (!_is_read) throw_no_read();
        return reinterpret_cast<const bool_t&>(_buffer[0]); 
    }

    size_t read()
    {
        _is_read = true;

        auto file_ptr = fopen_safe(_filename.c_str(), "rb");
        auto fp = file_ptr.get();
        std::fseek(fp, 0, SEEK_END);
        const size_t total_bytes = std::ftell(fp);

        _buffer.resize(total_bytes);
        std::fseek(fp, 0, SEEK_SET);
        const size_t read = std::fread(_buffer.data(), sizeof(char), _buffer.size(), fp);
        if (read != static_cast<size_t>(_buffer.size())) {
            throw std::runtime_error(
                "Could not read the whole file into buffer."
            );
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