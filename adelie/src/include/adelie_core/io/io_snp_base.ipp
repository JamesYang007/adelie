#pragma once
#include <adelie_core/io/io_snp_base.hpp>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#if defined(__linux__) 
#define MAP_FLAGS MAP_PRIVATE | MAP_NORESERVE | MAP_POPULATE
#elif defined(__APPLE__)
#define MAP_FLAGS MAP_PRIVATE
#else
#define MAP_FLAGS 0
#endif

namespace adelie_core {
namespace io {

ADELIE_CORE_IO_SNP_BASE_TP
size_t
ADELIE_CORE_IO_SNP_BASE::read()
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
        if (fd == -1) {
            perror("open");
            throw util::adelie_core_error("open failed.");
        }
        char* addr = static_cast<char*>(
            mmap(
                nullptr, 
                total_bytes, 
                PROT_READ,
                MAP_FLAGS,
                fd,
                0
            )
        );
        close(fd);
        if (addr == MAP_FAILED) {
            perror("mmap");
            throw util::adelie_core_error("mmap failed.");
        }
        _mmap_ptr = mmap_ptr_t(addr, [=](char* ptr) { munmap(ptr, total_bytes); } );
        new (&_buffer) Eigen::Map<buffer_t>(addr, total_bytes);
#else
        throw util::adelie_core_error("Only Linux and MacOS support the mmap feature.");
#endif
    // otherwise use the more general routine using file IO
    } else if (_read_mode == util::read_mode_type::_file) {
        _buffer_w.resize(total_bytes);
        const size_t read = std::fread(_buffer_w.data(), sizeof(char), _buffer_w.size(), fp);
        if (read != static_cast<size_t>(_buffer_w.size())) {
            throw util::adelie_core_error(
                "Could not read the whole file into buffer."
            );
        }
        new (&_buffer) Eigen::Map<buffer_t>(_buffer_w.data(), _buffer_w.size());

    } else {
        throw util::adelie_core_error("Unsupported read mode.");
    }

    if (endian() != is_big_endian()) {
        throw util::adelie_core_error(
            "Endianness is inconsistent! "
            "Regenerate the file on a machine with the same endianness."
        );
    }

    return total_bytes;
}

} // namespace io
} // namespace adelie_core

#undef MAP_FLAGS 