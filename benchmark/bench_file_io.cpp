#include <benchmark/benchmark.h>
#include <adelie_core/util/types.hpp>
#include <cstdio>
#include <memory>
#if defined(__linux__) || defined(__APPLE__)
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace ad = adelie_core;

static auto fopen_safe(
    const char* filename,
    const char* mode
)
{
    using file_unique_ptr_t = std::unique_ptr<
        std::FILE, 
        std::function<void(std::FILE*)>
    >;
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

static void BM_zero(benchmark::State& state)
{
    const auto n = state.range(0);
    ad::util::rowvec_type<char> buffer(n); 
    buffer.setZero();
    for (auto _ : state) {
        buffer.setZero();
        benchmark::DoNotOptimize(buffer);
    }
}

BENCHMARK(BM_zero)
    -> Args({10000000000}) // 10GB
    ;

static void BM_alloc(benchmark::State& state)
{
    const auto n = state.range(0);
    for (auto _ : state) {
        ad::util::rowvec_type<char> buffer(n); 
        buffer.setZero();
        buffer.setZero();
        benchmark::DoNotOptimize(buffer);
    }
}

BENCHMARK(BM_alloc)
    -> Args({10000000000}) // 10GB
    ;

static void BM_alloc_fragmented(benchmark::State& state)
{
    const auto n = state.range(0);
    for (auto _ : state) {
        constexpr auto size = 10;
        std::vector<ad::util::rowvec_type<char>> buffers(size); 
        for (auto& buffer : buffers) {
            buffer.resize(n / size);
            buffer.setZero();
        }
        benchmark::DoNotOptimize(buffers);
    }
}

BENCHMARK(BM_alloc_fragmented)
    -> Args({10000000000}) // 10GB
    ;

static void BM_fread(benchmark::State& state, bool buffered)
{
    const auto n = state.range(0);
    const char* filename = "/tmp/dummy";
    {
        ad::util::rowvec_type<char> buffer(n); 
        buffer.setZero();
        auto file = fopen_safe(filename, "wb");
        std::fwrite(buffer.data(), sizeof(char), n, file.get());
    }

    auto file = fopen_safe(filename, "rb");
    if (buffered) {
        std::setvbuf(file.get(), nullptr, _IOFBF, BUFSIZ);
    } else {
        std::setvbuf(file.get(), nullptr, _IONBF, 0);
    }
    ad::util::rowvec_type<char> buffer; 
    buffer.resize(n);
    for (auto _ : state) {
        std::fread(buffer.data(), sizeof(char), n, file.get()); 
        benchmark::DoNotOptimize(buffer);
        std::rewind(file.get());
    }

    std::remove(filename);    
}

static void BM_fread_buffered(benchmark::State& state) {
    BM_fread(state, true);
}

static void BM_fread_unbuffered(benchmark::State& state) {
    BM_fread(state, false);
}

BENCHMARK(BM_fread_buffered)
    -> Args({1000000000}) // 10GB
    ;
BENCHMARK(BM_fread_unbuffered)
    -> Args({1000000000}) // 10GB
    ;

static void BM_fread_read_copy(benchmark::State& state)
{
    const auto n = state.range(0);
    const char* filename = "/tmp/dummy";
    {
        ad::util::rowvec_type<char> buffer(n); 
        buffer.setZero();
        auto file = fopen_safe(filename, "wb");
        std::fwrite(buffer.data(), sizeof(char), n, file.get());
    }

    auto file = fopen_safe(filename, "rb");
    std::setvbuf(file.get(), nullptr, _IONBF, 0);
    ad::util::rowvec_type<char> buffer(n); 
    std::fread(buffer.data(), sizeof(char), n, file.get()); 
    ad::util::rowvec_type<char> buffer2(n);
    buffer2.setZero();
    for (auto _ : state) {
        const auto pos = rand() % n;
        const auto size = std::min<int>(n - pos, 1000000);
        buffer2.head(size) = Eigen::Map<const ad::util::rowvec_type<char>>(
            buffer.data() + pos, size
        );
        benchmark::DoNotOptimize(buffer2);
    }

    std::remove(filename);    
}

static void BM_mmap_read_copy(benchmark::State& state)
{
    const auto n = state.range(0);
    const char* filename = "/tmp/dummy";
    {
        ad::util::rowvec_type<char> buffer(n); 
        buffer.setZero();
        auto file = fopen_safe(filename, "wb");
        std::fwrite(buffer.data(), sizeof(char), n, file.get());
    }

    int fd = open(filename, O_RDONLY); 
    char* addr = (char*) mmap(
        nullptr,
        n,
        PROT_READ,
        MAP_PRIVATE,
        fd,
        0
    );
    close(fd);

    ad::util::rowvec_type<char> buffer(n);
    buffer.setZero();
    for (auto _ : state) {
        const auto pos = rand() % n;
        const auto size = std::min<int>(n - pos, 1000000);
        buffer.head(size) = Eigen::Map<const ad::util::rowvec_type<char>>(
            addr + pos, size
        );
        benchmark::DoNotOptimize(buffer);
    }

    munmap(addr, n);
    std::remove(filename);    
}

BENCHMARK(BM_fread_read_copy)
    -> Args({1000000000}) // 10GB
    ;
BENCHMARK(BM_mmap_read_copy)
    -> Args({1000000000}) // 10GB
    ;