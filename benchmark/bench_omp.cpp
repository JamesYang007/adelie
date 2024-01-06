#include <benchmark/benchmark.h>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
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


inline double cmul_par(
    const Eigen::Ref<const ad::util::rowvec_type<double>>& v1,
    const Eigen::Ref<const ad::util::rowvec_type<double>>& v2,
    size_t nt
)
{
    const auto n = v1.size();
    const size_t n_threads_cap = std::min<size_t>(nt, n);
    const int n_blocks = std::max<int>(n_threads_cap, 1);
    const int block_size = n / n_blocks;
    const int remainder = n % n_blocks;
    ad::util::rowvec_type<double> buff(n_blocks);
    #pragma omp parallel for schedule(static) num_threads(n_threads_cap)
    for (int t = 0; t < n_blocks; ++t)
    {
        const auto begin = (
            std::min<int>(t, remainder) * (block_size + 1) 
            + std::max<int>(t-remainder, 0) * block_size
        );
        const auto size = block_size + (t < remainder);
        buff[t] = v1.matrix().segment(begin, size).dot(v2.matrix().segment(begin, size));
    }
    return buff.sum();
}

static void BM_cmul_par(benchmark::State& state) {
    const auto n = state.range(0);
    const auto nt = state.range(1);

    ad::util::rowvec_type<double> v1(n); v1.setRandom();
    ad::util::rowvec_type<double> v2(n); v2.setRandom();
    double out;

    for (auto _ : state) {
        out = cmul_par(
            v1, v2, nt 
        );
        benchmark::DoNotOptimize(out);
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
        benchmark::DoNotOptimize(out);
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
    const auto nt = state.range(2);

    ad::util::colmat_type<double> m(n, p); m.setRandom();
    ad::util::rowvec_type<double> v(n); v.setRandom();
    ad::util::rowvec_type<double> out(p);

    for (auto _ : state) {
        const size_t n_threads_cap = nt;
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
            out.matrix().segment(begin, size) = v.matrix() * m.block(0, begin, n, size);
        }
    }
}

static void BM_bmul_par_cs(benchmark::State& state) {
    const auto n = state.range(0);
    const auto p = state.range(1);
    const auto nt = state.range(2);

    ad::util::colmat_type<double> m(n, p); m.setRandom();
    ad::util::rowvec_type<double> v(n); v.setRandom();
    ad::util::rowvec_type<double> out(p);
    
    const size_t n_threads_cap = std::min<size_t>(nt, n);
    const int n_blocks = std::max<int>(n_threads_cap, 1);
    const int block_size = n / n_blocks;
    const int remainder = n % n_blocks;

    ad::util::colmat_type<double> s(n_blocks, p);

    for (auto _ : state) {
        #pragma omp parallel num_threads(n_threads_cap)
        {
            #pragma omp for schedule(static) nowait
            for (int t = 0; t < n_blocks; ++t)
            {
                const auto begin = (
                    std::min<int>(t, remainder) * (block_size + 1) 
                    + std::max<int>(t-remainder, 0) * block_size
                );
                const auto size = block_size + (t < remainder);
                s.row(t).matrix().noalias() = (
                    v.matrix().segment(begin, size) * m.block(begin, 0, size, p)
                );
            }
        }
        out = s.rowwise().sum();
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

static void BM_svd(benchmark::State& state)
{
    const auto n = state.range(0);
    const auto p = state.range(1);
    ad::util::colmat_type<double> X(n, p);
    srand(0);
    X.setRandom();

    for (auto _ : state) {
        Eigen::BDCSVD<ad::util::colmat_type<double>> solver(
            X,
            Eigen::ComputeFullV
        );
        benchmark::DoNotOptimize(solver);
    }
}

static void BM_eigh(benchmark::State& state)
{
    const auto n = state.range(0);
    const auto p = state.range(1);
    ad::util::colmat_type<double> X(n, p);
    ad::util::colmat_type<double> XTX(p, p);
    srand(0);
    X.setRandom();

    Eigen::setNbThreads(8);
    for (auto _ : state) {
        XTX.noalias() = X.transpose() * X;
        Eigen::SelfAdjointEigenSolver<ad::util::colmat_type<double>> solver(XTX);
        benchmark::DoNotOptimize(solver);
    }
}

static void BM_keep_pool(benchmark::State& state)
{
    const auto n = state.range(0);
    int x = 0;
    for (auto _ : state) {
        for (int i = 0; i < 100; ++i) {
            #pragma omp parallel for schedule(static) num_threads(4)
            for (int i = 0; i < n; ++i) {
                x += i;
            }
            #pragma omp parallel for schedule(static) num_threads(4)
            for (int i = 0; i < n; ++i) {
                x += i;
            }
        }
        benchmark::DoNotOptimize(x);
    }
}

static void BM_change_pool(benchmark::State& state)
{
    const auto n = state.range(0);
    int x = 0;
    for (auto _ : state) {
        for (int i = 0; i < 100; ++i) {
            #pragma omp parallel for schedule(static) num_threads(4)
            for (int i = 0; i < n; ++i) {
                x += i;
            }
            #pragma omp parallel for schedule(static) num_threads(5)
            for (int i = 0; i < n; ++i) {
                x += i;
            }
        }
        benchmark::DoNotOptimize(x);
    }
}

static void BM_change_n_iters(benchmark::State& state)
{
    const auto n = state.range(0);
    int x = 0;
    srand(0);
    int m1 = rand() % n;
    int m2 = rand() % n;
    int m = (m1 + m2) / 2;
    for (auto _ : state) {
        for (int i = 0; i < 100; ++i) {
            #pragma omp parallel for schedule(static) num_threads(4)
            for (int j = 0; j < m1; ++j) {
                x += j;
            }
            #pragma omp parallel for schedule(static) num_threads(4)
            for (int j = 0; j < m2; ++j) {
                x += j;
            }
        }
        benchmark::DoNotOptimize(x);
    }
}

BENCHMARK(BM_cmul_seq)
    -> Args({10000000})
    ;
BENCHMARK(BM_cmul_par)
    -> Args({10000000, 1})
    -> Args({10000000, 2})
    -> Args({10000000, 4})
    -> Args({10000000, 8})
    -> Args({10000000, 16})
    ;
BENCHMARK(BM_ctmul_seq)
    -> Args({1000000})
    ;
BENCHMARK(BM_ctmul_par)
    -> Args({1000000, 1})
    -> Args({1000000, 2})
    -> Args({1000000, 4})
    -> Args({1000000, 8})
    -> Args({1000000, 16})
    ;

BENCHMARK(BM_bmul_seq)
    -> Args({1000000, 8})
    ;
BENCHMARK(BM_bmul_par_c)
    -> Args({1000000, 8, 1})
    -> Args({1000000, 8, 2})
    -> Args({1000000, 8, 4})
    -> Args({1000000, 8, 8})
    -> Args({1000000, 8, 16})
    ;
BENCHMARK(BM_bmul_par_cs)
    -> Args({500000, 8, 1})
    -> Args({500000, 8, 2})
    -> Args({500000, 8, 4})
    -> Args({500000, 8, 8})
    -> Args({500000, 8, 16})
    ;
BENCHMARK(BM_btmul_seq)
    -> Args({1000000, 8})
    ;
BENCHMARK(BM_btmul_par_c)
    -> Args({1000000, 8, 2})
    -> Args({1000000, 8, 4})
    -> Args({1000000, 8, 8})
    ;

BENCHMARK(BM_svd)
    -> Args({100, 8})
    -> Args({1000, 8})
    -> Args({10000, 8})
    -> Args({100000, 8})
    ;
BENCHMARK(BM_eigh)
    -> Args({100, 8})
    -> Args({1000, 8})
    -> Args({10000, 8})
    -> Args({100000, 8})
    ;
BENCHMARK(BM_keep_pool)
    -> Args({1000})
    ;
BENCHMARK(BM_change_pool)
    -> Args({1000})
    ;
BENCHMARK(BM_change_n_iters)
    -> Args({1000})
    ;