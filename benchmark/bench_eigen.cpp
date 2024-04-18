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

static void BM_ctmul(benchmark::State& state) {
    const auto n = state.range(0);
    ad::util::rowvec_type<double> x(n);
    x.setRandom();
    ad::util::rowvec_type<double> y(n);
    y.setRandom();

    for (auto _ : state) {
        y -= x[0] * x;
        benchmark::DoNotOptimize(y);
    }
}

BENCHMARK(BM_ctmul)
    -> Args({10})
    -> Args({100})
    -> Args({1000})
    -> Args({10000})
    -> Args({100000})
    ;

static void BM_ctmul_subi(benchmark::State& state) {
    const auto n = state.range(0);
    ad::util::rowvec_type<double> x(n);
    x.setRandom();
    ad::util::rowvec_type<double> y(n);
    y.setRandom();
    ad::util::rowvec_type<double> out(n);

    for (auto _ : state) {
        out = x[0] * x;
        y -= out;
        benchmark::DoNotOptimize(out);
        benchmark::DoNotOptimize(y);
    }
}

BENCHMARK(BM_ctmul_subi)
    -> Args({10})
    -> Args({100})
    -> Args({1000})
    -> Args({10000})
    -> Args({100000})
    ;

static void BM_btmul(benchmark::State& state) {
    const auto n = state.range(0);
    const auto p = 10;
    ad::util::colmat_type<double> x(n, p);
    x.setRandom();
    ad::util::rowvec_type<double> a(p);
    a.setRandom();
    ad::util::rowvec_type<double> y(n);
    y.setRandom();

    for (auto _ : state) {
        y.matrix() -= a.matrix() * x.matrix().transpose();
        benchmark::DoNotOptimize(y);
    }
}

BENCHMARK(BM_btmul)
    -> Args({10})
    -> Args({100})
    -> Args({1000})
    -> Args({10000})
    -> Args({100000})
    ;

static void BM_btmul_subi(benchmark::State& state) {
    const auto n = state.range(0);
    const auto p = 10;
    ad::util::colmat_type<double> x(n, p);
    x.setRandom();
    ad::util::rowvec_type<double> a(p);
    a.setRandom();
    ad::util::rowvec_type<double> out(n);
    ad::util::rowvec_type<double> y(n);
    y.setRandom();

    for (auto _ : state) {
        out.matrix() = a.matrix() * x.matrix().transpose();
        y -= out;
        benchmark::DoNotOptimize(y);
        benchmark::DoNotOptimize(out);
    }
}

BENCHMARK(BM_btmul_subi)
    -> Args({10})
    -> Args({100})
    -> Args({1000})
    -> Args({10000})
    -> Args({100000})
    ;

static void BM_spdot(benchmark::State& state) {
    const auto n = state.range(0);
    ad::util::rowvec_type<int> inner = ad::util::rowvec_type<int>::LinSpaced(n, 0, n-1);
    ad::util::rowvec_type<int> value(n);
    constexpr int max_val = 2;
    value.setRandom();
    value = value.unaryExpr([&](auto x) { return std::abs(x) % max_val + 1; });
    ad::util::rowvec_type<double> v(n);
    v.setRandom();
    double sum = 0;

    for (auto _ : state) {
        for (int i = 0; i < inner.size(); ++i) {
            sum += value[i] * v[inner[i]];
        } 
        benchmark::DoNotOptimize(sum);
    }
}

BENCHMARK(BM_spdot)
    -> Args({10})
    -> Args({100})
    -> Args({1000})
    -> Args({10000})
    -> Args({100000})
    ;

static void BM_spdot_cached(benchmark::State& state) {
    const auto n = state.range(0);
    ad::util::rowvec_type<int> inner = ad::util::rowvec_type<int>::LinSpaced(n, 0, n-1);
    ad::util::rowvec_type<int> value(n);
    constexpr int max_val = 2;
    value.setRandom();
    value = value.unaryExpr([&](auto x) { return std::abs(x) % max_val + 1; });
    ad::util::rowvec_type<double> v(n);
    v.setRandom();
    ad::util::rowvec_type<double, max_val> cache;
    double sum = 0;

    for (auto _ : state) {
        cache.setZero();
        for (int i = 0; i < inner.size(); ++i) {
            if (value[i] == 1) {
                cache[0] += v[inner[i]];
            } else {
                cache[1] += v[inner[i]];
            }
        } 
        sum = cache[0] + 2 * cache[1];
        benchmark::DoNotOptimize(sum);
    }
}

BENCHMARK(BM_spdot_cached)
    -> Args({10})
    -> Args({100})
    -> Args({1000})
    -> Args({10000})
    -> Args({100000})
    ;

static void BM_spaddi(benchmark::State& state) {
    const auto n = state.range(0);
    ad::util::rowvec_type<int> inner = ad::util::rowvec_type<int>::LinSpaced(n, 0, n-1);
    ad::util::rowvec_type<int> value(n);
    constexpr int max_val = 2;
    value.setRandom();
    value = value.unaryExpr([&](auto x) { return std::abs(x) % max_val + 1; });
    ad::util::rowvec_type<double> v(n);
    v.setRandom();
    double a = n;

    for (auto _ : state) {
        for (int i = 0; i < inner.size(); ++i) {
            v[inner[i]] += value[i] * a;
        } 
        benchmark::DoNotOptimize(v);
    }
}

BENCHMARK(BM_spaddi)
    -> Args({10})
    -> Args({100})
    -> Args({1000})
    -> Args({10000})
    -> Args({100000})
    ;

static void BM_spaddi_branch(benchmark::State& state) {
    const auto n = state.range(0);
    ad::util::rowvec_type<int> inner = ad::util::rowvec_type<int>::LinSpaced(n, 0, n-1);
    ad::util::rowvec_type<int> value(n);
    constexpr int max_val = 2;
    value.setRandom();
    value = value.unaryExpr([&](auto x) { return std::abs(x) % max_val + 1; });
    ad::util::rowvec_type<double> v(n);
    v.setRandom();
    double a = n;
    double a2 = 2 * a;

    for (auto _ : state) {
        for (int i = 0; i < inner.size(); ++i) {
            v[inner[i]] += (value[i] == 1) ? a : a2;
            //if (value[i] == 1) {
            //    v[inner[i]] += a;
            //} else {
            //    v[inner[i]] += a2;
            //}
        } 
        benchmark::DoNotOptimize(v);
    }
}

BENCHMARK(BM_spaddi_branch)
    -> Args({10})
    -> Args({100})
    -> Args({1000})
    -> Args({10000})
    -> Args({100000})
    ;

static void BM_bmul(benchmark::State& state) {
    const auto n = state.range(0);
    const auto p = 10;
    ad::util::colmat_type<double> x(n, p);
    x.setRandom();
    ad::util::rowvec_type<double> v(n);
    v.setRandom();
    ad::util::rowvec_type<double> w(n);
    w.setRandom();
    ad::util::rowvec_type<double> out(p);

    for (auto _ : state) {
        out.matrix().noalias() = (v*w).matrix() * x;
        benchmark::DoNotOptimize(out);
    }
}

BENCHMARK(BM_bmul)
    -> Args({10})
    -> Args({100})
    -> Args({1000})
    -> Args({10000})
    -> Args({100000})
    ;

static void BM_bmul_cached(benchmark::State& state) {
    const auto n = state.range(0);
    const auto p = 10;
    ad::util::colmat_type<double> x(n, p);
    x.setRandom();
    ad::util::rowvec_type<double> v(n);
    v.setRandom();
    ad::util::rowvec_type<double> w(n);
    w.setRandom();
    ad::util::rowvec_type<double> buff(n);
    ad::util::rowvec_type<double> out(p);

    for (auto _ : state) {
        buff = v * w;
        out.matrix().noalias() = buff.matrix() * x;
        benchmark::DoNotOptimize(buff);
        benchmark::DoNotOptimize(out);
    }
}

BENCHMARK(BM_bmul_cached)
    -> Args({10})
    -> Args({100})
    -> Args({1000})
    -> Args({10000})
    -> Args({100000})
    ;