#include <benchmark/benchmark.h>
#include <adelie_core/util/types.hpp>
#include <omp.h>
#include <iostream>

namespace ad = adelie_core;

static void BM_cmul_seq(benchmark::State& state) {
    const auto n = state.range(0);

    ad::util::rowvec_type<double> v1(n); v1.setRandom();
    ad::util::rowvec_type<double> v2(n); v2.setRandom();
    double out;

    for (auto _ : state) {
        out = v1.matrix().dot(v2.matrix());
        benchmark::DoNotOptimize(out);
    }
}

static void BM_cmul_par(benchmark::State& state) {
    const auto n = state.range(0);
    const auto nt = state.range(1);

    ad::util::rowvec_type<double> v1(n); v1.setRandom();
    ad::util::rowvec_type<double> v2(n); v2.setRandom();
    double out;

    for (auto _ : state) {
        const size_t n_threads_cap = std::min<size_t>(nt, n);
        const int n_blocks = std::max<int>(n_threads_cap, 1);
        const int block_size = n / n_blocks;
        const int remainder = n % n_blocks;
        #pragma omp parallel for schedule(static) num_threads(n_threads_cap) reduction(+:out)
        for (int t = 0; t < n_blocks; ++t)
        {
            const auto begin = (
                std::min<int>(t, remainder) * (block_size + 1) 
                + std::max<int>(t-remainder, 0) * block_size
            );
            const auto size = block_size + (t < remainder);
            out += v1.matrix().segment(begin, size).dot(v2.matrix().segment(begin, size));
        }
        out = 0;
    }
}

static void BM_ctmul_seq(benchmark::State& state) {
    const auto n = state.range(0);

    double c = 3.14;
    ad::util::rowvec_type<double> v(n); v.setRandom();
    ad::util::rowvec_type<double> out(n);

    for (auto _ : state) {
        out = c * v;
        benchmark::DoNotOptimize(out);
    }
}

static void BM_ctmul_par(benchmark::State& state) {
    const auto n = state.range(0);
    const auto nt = state.range(1);

    double c = 3.14;
    ad::util::rowvec_type<double> v(n); v.setRandom();
    ad::util::rowvec_type<double> out(n);

    for (auto _ : state) {
        const size_t n_threads_cap = std::min<size_t>(nt, out.size());
        const int n_blocks = std::max<int>(n_threads_cap, 1);
        const int block_size = out.size() / n_blocks;
        const int remainder = out.size() % n_blocks;
        #pragma omp parallel for schedule(static) num_threads(n_threads_cap)
        for (int t = 0; t < n_blocks; ++t)
        {
            const auto begin = (
                std::min<int>(t, remainder) * (block_size + 1) 
                + std::max<int>(t-remainder, 0) * block_size
            );
            const auto size = block_size + (t < remainder);
            out.matrix().segment(begin, size) = c * v.segment(begin, size);
        }
    }
}

static void BM_bmul_seq(benchmark::State& state) {
    const auto n = state.range(0);
    const auto p = state.range(1);

    ad::util::colmat_type<double> m(n, p); m.setRandom();
    ad::util::rowvec_type<double> v(n); v.setRandom();
    ad::util::rowvec_type<double> out(p);

    for (auto _ : state) {
        out.matrix().noalias() = v.matrix() * m;
    }
}

static void BM_bmul_par_c(benchmark::State& state) {
    const auto n = state.range(0);
    const auto p = state.range(1);
    const auto nt = std::min<int>(state.range(2), p);

    ad::util::colmat_type<double> m(n, p); m.setRandom();
    ad::util::rowvec_type<double> v(n); v.setRandom();
    ad::util::rowvec_type<double> out(p);

    for (auto _ : state) {
        #pragma omp parallel for schedule(static) num_threads(nt)
        for (int i = 0; i < p; ++i)
        {
            out[i] = v.matrix() * m.col(i);
        }
    }
}

static void BM_btmul_seq(benchmark::State& state) {
    const auto n = state.range(0);
    const auto p = state.range(1);

    ad::util::colmat_type<double> m(n, p); m.setRandom();
    ad::util::rowvec_type<double> v(p); v.setRandom();
    ad::util::rowvec_type<double> out(n);

    Eigen::setNbThreads(15);

    for (auto _ : state) {
        out.matrix().noalias() = v.matrix() * m.transpose();
    }
}

static void BM_btmul_par_c(benchmark::State& state) {
    const auto n = state.range(0);
    const auto p = state.range(1);
    const auto nt = std::min<int>(state.range(2), n);

    ad::util::colmat_type<double> m(n, p); m.setRandom();
    ad::util::rowvec_type<double> v(p); v.setRandom();
    ad::util::rowvec_type<double> out(n);

    for (auto _ : state) {
        const size_t n_threads_cap = std::min<size_t>(nt, out.size());
        const int n_blocks = std::max<int>(n_threads_cap, 1);
        const int block_size = out.size() / n_blocks;
        const int remainder = out.size() % n_blocks;
        #pragma omp parallel for schedule(static) num_threads(n_threads_cap)
        for (int t = 0; t < n_blocks; ++t)
        {
            const auto begin = (
                std::min<int>(t, remainder) * (block_size + 1) 
                + std::max<int>(t-remainder, 0) * block_size
            );
            const auto size = block_size + (t < remainder);

            out.matrix().segment(begin, size).noalias() = (
                v.matrix() * m.block(begin, 0, size, p).transpose()
            );
        }
    }
}

BENCHMARK(BM_cmul_seq)
    -> Args({1000000})
    ;
BENCHMARK(BM_cmul_par)
    -> Args({1000000, 4})
    -> Args({1000000, 8})
    -> Args({1000000, 15})
    ;
BENCHMARK(BM_ctmul_seq)
    -> Args({1000000})
    ;
BENCHMARK(BM_ctmul_par)
    -> Args({1000000, 4})
    -> Args({1000000, 8})
    -> Args({1000000, 15})
    ;

BENCHMARK(BM_bmul_seq)
    -> Args({1000000, 8})
    ;
BENCHMARK(BM_bmul_par_c)
    -> Args({1000000, 8, 4})
    -> Args({1000000, 8, 8})
    -> Args({1000000, 8, 15})
    ;
BENCHMARK(BM_btmul_seq)
    -> Args({1000000, 4})
    ;
BENCHMARK(BM_btmul_par_c)
    -> Args({1000000, 4, 2})
    -> Args({1000000, 4, 4})
    -> Args({1000000, 4, 8})
    -> Args({1000000, 4, 15})
    ;