#include <benchmark/benchmark.h>
#include <adelie_core/util/types.hpp>
#include <omp.h>
#include <iostream>

namespace ad = adelie_core;

static void BM_copy_mul(benchmark::State& state) {
    const auto n = state.range(0);
    const auto p = 1;
    const auto K = 4;
    ad::util::colmat_type<double> X(n, p); X.setRandom();
    ad::util::rowmat_type<double> V(n, K); V.setRandom();
    ad::util::rowvec_type<double> v(n);
    ad::util::rowvec_type<double> out(p);

    for (auto _ : state) {
        v = V.col(0);
        benchmark::DoNotOptimize(v);
        out = v.matrix() * X;
        benchmark::DoNotOptimize(out);
    }
}

BENCHMARK(BM_copy_mul)
    -> Args({10})
    -> Args({100})
    -> Args({1000})
    -> Args({10000})
    -> Args({100000})
    ;

static void BM_mul(benchmark::State& state) {
    const auto n = state.range(0);
    const auto p = 1;
    const auto K = 4;
    ad::util::colmat_type<double> X(n, p); X.setRandom();
    ad::util::rowmat_type<double> V(n, K); V.setRandom();
    ad::util::rowvec_type<double> out(p);

    for (auto _ : state) {
        out = V.col(0).transpose() * X;
        benchmark::DoNotOptimize(out);
    }
}

BENCHMARK(BM_mul)
    -> Args({10})
    -> Args({100})
    -> Args({1000})
    -> Args({10000})
    -> Args({100000})
    ;

static void BM_copy(benchmark::State& state) {
    const auto n = state.range(0);
    const auto K = 4;
    ad::util::rowmat_type<double> V(n, K); V.setRandom();
    ad::util::rowvec_type<double> v(n);

    for (auto _ : state) {
        v = V.col(0);
        benchmark::DoNotOptimize(v);
    }
}

BENCHMARK(BM_copy)
    -> Args({10})
    -> Args({100})
    -> Args({1000})
    -> Args({10000})
    -> Args({100000})
    ;

static void BM_dot(benchmark::State& state) {
    const auto n = state.range(0);
    ad::util::rowvec_type<double> x(n);
    x.setRandom();
    ad::util::rowvec_type<double> y(n);
    y.setRandom();
    ad::util::rowvec_type<double> z(n);
    z.setRandom();

    for (auto _ : state) {
        auto out = (x * y * z).sum();
        benchmark::DoNotOptimize(out);
    }
}

BENCHMARK(BM_dot)
    -> Args({10})
    -> Args({100})
    -> Args({1000})
    -> Args({10000})
    -> Args({100000})
    ;