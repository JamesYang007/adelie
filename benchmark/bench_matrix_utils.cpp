#include <benchmark/benchmark.h>
#include <adelie_core/matrix/utils.hpp>
#include <adelie_core/util/types.hpp>

namespace ad = adelie_core;

static void BM_dvaddi(benchmark::State& state) {
    const auto nt = state.range(0);
    const auto n = 250000;
    ad::util::rowvec_type<double> v1(n); v1.setRandom();
    ad::util::rowvec_type<double> v2(n); v2.setRandom();

    for (auto _ : state) {
        ad::matrix::dvaddi(v1, v2, nt);
        benchmark::DoNotOptimize(v1);
    }
}

BENCHMARK(BM_dvaddi)
    -> Arg(1)
    -> Arg(2)
    -> Arg(4)
    -> Arg(8)
    -> Arg(64)
    ;

static void BM_dvsubi(benchmark::State& state) {
    const auto nt = state.range(0);
    const auto n = 1000000;
    ad::util::rowvec_type<double> v1(n); v1.setRandom();
    ad::util::rowvec_type<double> v2(n); v2.setRandom();

    for (auto _ : state) {
        ad::matrix::dvsubi(v1, v2, nt);
        benchmark::DoNotOptimize(v1);
    }
}

BENCHMARK(BM_dvsubi)
    -> Arg(1)
    -> Arg(2)
    -> Arg(4)
    -> Arg(8)
    -> Arg(64)
    ;