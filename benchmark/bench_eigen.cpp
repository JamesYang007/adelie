#include <benchmark/benchmark.h>
#include <adelie_core/util/types.hpp>
#include <omp.h>
#include <iostream>
#include <numeric>

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
    constexpr int max_val = 3;
    value.setRandom();
    value = value.unaryExpr([&](auto x) { return std::abs(x) % max_val + 1; });
    ad::util::rowvec_type<double> v(n);
    v.setRandom();
    double sum = 0;
    ad::util::rowvec_type<double, 3> cache;
    cache.setZero();

    for (auto _ : state) {
        for (int i = 0; i < inner.size(); ++i) {
            cache[value[i]] += v[inner[i]];
        } 
        sum += cache.sum();
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
    constexpr int max_val = 3;
    value.setRandom();
    value = value.unaryExpr([&](auto x) { return std::abs(x) % max_val + 1; });
    ad::util::rowvec_type<double> v(n);
    v.setRandom();
    ad::util::rowvec_type<double, max_val> cache;
    double sum = 0;

    for (auto _ : state) {
        cache.setZero();
        for (int i = 0; i < inner.size(); ++i) {
            if (value[i] < 0) {
                cache[0] += v[inner[i]];
            } else if (value[i] == 1) {
                cache[1] += v[inner[i]];
            } else {
                cache[2] += v[inner[i]];
            }
        } 
        sum = cache[0] + 2 * cache[1] + 3 * cache[2];
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

static void BM_dot_1prod(benchmark::State& state) {
    const auto n = state.range(0);
    ad::util::colmat_type<double> x(n, 100);
    x.setRandom();
    ad::util::rowvec_type<double> v(n);
    v.setRandom();
    double sum = 0;

    for (auto _ : state) {
        sum += x.col(50).dot(v.matrix());
        benchmark::DoNotOptimize(sum);
    }
}

BENCHMARK(BM_dot_1prod)
    -> Args({10})
    -> Args({100})
    -> Args({1000})
    -> Args({10000})
    -> Args({100000})
    ;

static void BM_dot_2prod(benchmark::State& state) {
    const auto n = state.range(0);
    ad::util::colmat_type<double> x(n, 100);
    x.setRandom();
    ad::util::rowvec_type<double> v(n);
    v.setRandom();
    ad::util::rowvec_type<double> w(n);
    w.setRandom();
    double sum = 0;

    for (auto _ : state) {
        sum += x.col(50).dot((v*w).matrix());
        benchmark::DoNotOptimize(sum);
    }
}

BENCHMARK(BM_dot_2prod)
    -> Args({10})
    -> Args({100})
    -> Args({1000})
    -> Args({10000})
    -> Args({100000})
    ;

static void BM_sparse_split(benchmark::State& state) {
    const auto n = state.range(0);
    const auto k = n / 100;
    srand(0);
    ad::util::rowvec_type<uint32_t> inner1(k); inner1.setRandom();
    inner1 = inner1.unaryExpr([&](auto i) -> uint32_t { return i % n; });
    ad::util::rowvec_type<uint32_t> inner2(k); inner2.setRandom();
    inner2 = inner2.unaryExpr([&](auto i) -> uint32_t { return i % n; });
    ad::util::rowvec_type<uint32_t> inner3(k); inner3.setRandom();
    inner3 = inner3.unaryExpr([&](auto i) -> uint32_t { return i % n; });
    std::sort(inner1.data(), inner1.data() + k);
    std::sort(inner2.data(), inner2.data() + k);
    std::sort(inner3.data(), inner3.data() + k);
    ad::util::rowvec_type<double> v(n);
    v.setRandom();
    double sum = 0;

    for (auto _ : state) {
        for (int i = 0; i < inner1.size(); ++i) {
            sum += v[inner1[i]];
        }
        for (int i = 0; i < inner2.size(); ++i) {
            sum += 2 * v[inner2[i]];
        }
        for (int i = 0; i < inner3.size(); ++i) {
            sum += 2.123 * v[inner3[i]];
        }
        benchmark::DoNotOptimize(sum);
    }
}

BENCHMARK(BM_sparse_split)
    -> Args({100})
    -> Args({1000})
    -> Args({10000})
    -> Args({100000})
    ;

static void BM_sparse_combine(benchmark::State& state) {
    const auto n = state.range(0);
    const auto k = n / 100;
    srand(0);
    ad::util::rowvec_type<uint32_t> inner1(k); inner1.setRandom();
    inner1 = inner1.unaryExpr([&](auto i) -> uint32_t { return i % n; });
    ad::util::rowvec_type<uint32_t> inner2(k); inner2.setRandom();
    inner2 = inner2.unaryExpr([&](auto i) -> uint32_t { return i % n; });
    ad::util::rowvec_type<uint32_t> inner3(k); inner3.setRandom();
    inner3 = inner3.unaryExpr([&](auto i) -> uint32_t { return i % n; });
    ad::util::rowvec_type<uint32_t> inner(3 * k);
    inner.segment(0, k) = inner1;
    inner.segment(k, k) = inner2;
    inner.segment(2 * k, k) = inner3;
    std::sort(inner.data(), inner.data() + 3 * k);
    ad::util::rowvec_type<int8_t> value(3 * k); value.setRandom();
    value = value.unaryExpr([&](auto i) -> int8_t { return i % 2 + 1; });
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

BENCHMARK(BM_sparse_combine)
    -> Args({100})
    -> Args({1000})
    -> Args({10000})
    -> Args({100000})
    ;

static void BM_assign_eigen(benchmark::State& state) {
    const auto n = state.range(0);
    ad::util::rowvec_type<double> x(n);
    x.setRandom();
    ad::util::rowvec_type<double> y(n);

    for (auto _ : state) {
        y = x;
        benchmark::DoNotOptimize(y);
    }
}

BENCHMARK(BM_assign_eigen)
    -> Args({100})
    -> Args({1000})
    -> Args({10000})
    -> Args({100000})
    ;

static void BM_assign_manual(benchmark::State& state) {
    const auto n = state.range(0);
    ad::util::rowvec_type<double> x(n);
    x.setRandom();
    ad::util::rowvec_type<int> indices(n);
    std::iota(indices.data(), indices.data()+n, 0);
    ad::util::rowvec_type<double> y(n);

    for (auto _ : state) {
        for (int i = 0; i < n; ++i) y[indices[i]] = x[indices[i]];
        benchmark::DoNotOptimize(y);
    }
}

static void BM_mask(benchmark::State& state) {
    const auto n = state.range(0);
    ad::util::rowvec_type<double> x(n);
    x.setRandom();
    ad::util::rowvec_type<double> y(n);
    y.setRandom();
    y = y.unaryExpr([&](auto i) -> double { return i > 0; });
    ad::util::rowvec_type<double> z(n);

    for (auto _ : state) {
        z = x * y;
        benchmark::DoNotOptimize(z);
    }
}

BENCHMARK(BM_mask)
    -> Args({100})
    -> Args({1000})
    -> Args({10000})
    -> Args({100000})
    ;

static void BM_one_scan(benchmark::State& state)
{
    const auto n = state.range(0);
    const auto k = state.range(1);
    ad::util::rowvec_type<uint32_t> x(n);
    x.setRandom();
    x = x.unaryExpr([&](auto i) -> uint32_t { return i % k; });
    ad::util::rowvec_type<double> y(n);
    ad::util::rowvec_type<double> out(k);

    for (auto _ : state) {
        out.setZero();
        for (int i = 0; i < x.size(); ++i) {
            out[x[i]] += y[i];
        }
    }
}

BENCHMARK(BM_one_scan)
    -> Args({100, 2})
    -> Args({1000, 2})
    -> Args({10000, 2})
    -> Args({100000, 2})
    -> Args({100, 10})
    -> Args({1000, 10})
    -> Args({10000, 10})
    -> Args({100000, 10})
    ;

static void BM_many_scan(benchmark::State& state)
{
    const auto n = state.range(0);
    const auto k = state.range(1);
    ad::util::rowvec_type<uint32_t> x(n);
    x.setRandom();
    x = x.unaryExpr([&](auto i) -> uint32_t { return i % k; });
    ad::util::rowvec_type<double> y(n);
    ad::util::rowvec_type<double> out(k);

    for (auto _ : state) {
        for (int j = 0; j < k; ++j) {
            double sum = 0;
            for (int i = 0; i < x.size(); ++i) {
                if (x[i] == j) sum += y[i];
            }
            out[j] = sum;
        }
    }
}

BENCHMARK(BM_many_scan)
    -> Args({100, 2})
    -> Args({1000, 2})
    -> Args({10000, 2})
    -> Args({100000, 2})
    -> Args({100, 10})
    -> Args({1000, 10})
    -> Args({10000, 10})
    -> Args({100000, 10})
    ;

static void BM_xtx_sym(benchmark::State& state)
{
    const auto n = state.range(0);
    const auto d = 100;
    ad::util::colmat_type<double> X;
    X.setRandom(n, d);
    ad::util::colmat_type<double> XTX;
    XTX.resize(d, d);

    for (auto _ : state) {
        XTX.setZero();
        XTX.template selfadjointView<Eigen::Lower>().rankUpdate(X.transpose());
        XTX.template triangularView<Eigen::Upper>() = XTX.transpose();
    }
}

BENCHMARK(BM_xtx_sym)
    -> Args({10000})
    -> Args({100000})
    -> Args({1000000})
    ;

static void BM_xtx_naive(benchmark::State& state)
{
    const auto n = state.range(0);
    const auto n_threads = state.range(1);
    const auto d = 100;
    ad::util::colmat_type<double> X;
    X.setRandom(n, d);
    ad::util::colmat_type<double> XTX;
    XTX.resize(d, d);

    Eigen::setNbThreads(n_threads);

    for (auto _ : state) {
        XTX.noalias() = X.transpose() * X;
    }
}

BENCHMARK(BM_xtx_naive)
    -> Args({10000, 1})
    -> Args({100000, 1})
    -> Args({1000000, 1})
    -> Args({10000, 4})
    -> Args({100000, 4})
    -> Args({1000000, 4})
    -> Args({10000, 8})
    -> Args({100000, 8})
    -> Args({1000000, 8})
    -> Args({10000, 16})
    -> Args({100000, 16})
    -> Args({1000000, 16})
    ;

static void BM_xtwx_naive(benchmark::State& state)
{
    const auto n = state.range(0);
    const auto n_threads = state.range(1);
    const auto d = 100;
    ad::util::colmat_type<double> X;
    X.setRandom(n, d);
    ad::util::colvec_type<double> w;
    w.setRandom(n);
    ad::util::colvec_type<bool> mask;
    mask.setRandom(n);
    ad::util::colmat_type<double> XTX;
    XTX.resize(d, d);

    Eigen::setNbThreads(n_threads);

    for (auto _ : state) {
        XTX.noalias() = X.transpose() * (w.square() * mask.template cast<double>()).matrix().asDiagonal() * X;
    }
}

BENCHMARK(BM_xtwx_naive)
    -> Args({10000, 1})
    -> Args({100000, 1})
    -> Args({1000000, 1})
    -> Args({10000, 4})
    -> Args({100000, 4})
    -> Args({1000000, 4})
    -> Args({10000, 8})
    -> Args({100000, 8})
    -> Args({1000000, 8})
    ;

static void BM_xtwx_naive_buffer(benchmark::State& state)
{
    const auto n = state.range(0);
    const auto n_threads = state.range(1);
    const auto d = 100;
    ad::util::colmat_type<double> X;
    X.setRandom(n, d);
    ad::util::colvec_type<double> w;
    w.setRandom(n);
    ad::util::colvec_type<bool> mask;
    mask.setRandom(n);
    ad::util::colmat_type<double> buff(n, d);
    ad::util::colmat_type<double> XTX;
    XTX.resize(d, d);

    Eigen::setNbThreads(n_threads);

    for (auto _ : state) {
        buff.array() = X.array().colwise() * (w * mask.template cast<double>());
        XTX.noalias() = buff.transpose() * buff;
    }
}

BENCHMARK(BM_xtwx_naive_buffer)
    -> Args({10000, 1})
    -> Args({100000, 1})
    -> Args({1000000, 1})
    -> Args({10000, 4})
    -> Args({100000, 4})
    -> Args({1000000, 4})
    -> Args({10000, 8})
    -> Args({100000, 8})
    -> Args({1000000, 8})
    ;

static void BM_matrix_vector_dense(benchmark::State& state)
{
    const auto n = state.range(0);
    const auto d = 10;
    ad::util::colmat_type<double> X(d, n);
    ad::util::colvec_type<double> v(n);
    X.setRandom();
    for (int i = 0; i < n; i += (n / 100)) {
        v[i] = 1;
    }
    ad::util::colvec_type<double> out(d);

    for (auto _ : state) {
        out.matrix() = X * v.matrix();
    }
}

BENCHMARK(BM_matrix_vector_dense)
    -> Args({1000})
    -> Args({10000})
    -> Args({100000})
    ;

static void BM_matrix_vector_sparse(benchmark::State& state)
{
    const auto n = state.range(0);
    const auto d = 10;
    ad::util::colmat_type<double> X(d, n);
    ad::util::colvec_type<double> v(n);
    X.setRandom();
    for (int i = 0; i < n; i += (n / 1000)) {
        v[i] = 1;
    }
    ad::util::colvec_type<double> out(d);

    for (auto _ : state) {
        out.setZero();
        //for (int i = 0; i < n; i += (n / 1000)) {
        //    out += v[i] * X.col(i).array();
        //}
        for (int i = 0; i < n; ++i) {
            if (i % (n / 1000)) continue;
            out += v[i] * X.col(i).array();
        }
        benchmark::DoNotOptimize(out);
    }
}

BENCHMARK(BM_matrix_vector_sparse)
    -> Args({1000})
    -> Args({10000})
    -> Args({100000})
    ;
