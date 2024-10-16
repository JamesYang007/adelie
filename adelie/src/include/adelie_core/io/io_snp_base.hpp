#pragma once
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <adelie_core/util/exceptions.hpp>
#include <adelie_core/util/types.hpp>

namespace adelie_core {
namespace io {

template <class MmapPtrType=std::unique_ptr<char, std::function<void(char*)>>>
class IOSNPBase
{
public:
    using string_t = std::string;
    using file_unique_ptr_t = std::unique_ptr<
        std::FILE, 
        std::function<void(std::FILE*)>
    >;
    using mmap_ptr_t = MmapPtrType;
    using bool_t = bool;
    using buffer_t = util::rowvec_type<char>;

protected:
    const string_t _filename;
    const util::read_mode_type _read_mode;
    buffer_t _buffer_w;             // only used when _read_mode == _file
    mmap_ptr_t _mmap_ptr;           // only used when _read_mode == _mmap 
    Eigen::Map<buffer_t> _buffer;
    bool_t _is_read;

    static void throw_no_read() 
    {
        throw util::adelie_core_error(
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
            throw util::adelie_core_error("Cannot open file " + std::string(filename));
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
    explicit IOSNPBase(
        const string_t& filename,
        const string_t& read_mode
    ):
        _filename(filename),
        _read_mode(convert_read_mode(read_mode)),
        _mmap_ptr(nullptr),
        _buffer(nullptr, 0),
        _is_read(false)
    {}

    virtual ~IOSNPBase() {}

    bool_t is_read() const { return _is_read; }

    bool_t endian() const { 
        if (!_is_read) throw_no_read();
        return reinterpret_cast<const bool_t&>(_buffer[0]); 
    }

    virtual size_t read();
};

} // namespace io
} // namespace adelie_core