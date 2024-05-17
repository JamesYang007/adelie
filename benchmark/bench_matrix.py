from adelie.adelie_core import matrix
import adelie as ad
import matplotlib.pyplot as plt
import numpy as np


def bench_dv(
    method,
    n_list=None, 
    n_threads_list=None,
    n_sims=10,
    min_flops=0,
):
    fun = {
        "dvaddi": matrix.bench_dvaddi,
        "dvsubi": matrix.bench_dvsubi,
        "dvveq": matrix.bench_dvveq,
        "dvzero": matrix.bench_dvzero,
        "ddot": matrix.bench_ddot,
    }[method]

    ad.configs.set_configs("min_flops", min_flops)
    if n_list is None:
        n_list = 2 ** np.arange(0, 24)
    if n_threads_list is None:
        n_threads_list = 2 ** np.arange(0, 4)
    out = np.empty((len(n_threads_list), len(n_list)))
    n_max = np.max(n_list)
    x_max = np.random.normal(0, 1, n_max)
    y_max = np.random.normal(0, 1, n_max)
    for i, n_threads in enumerate(n_threads_list):
        for j, n in enumerate(n_list):
            x = x_max[:n]
            y = y_max[:n]
            out[i,j] = fun(x, y, n_threads, n_sims)

    fig, ax = plt.subplots(1, 1, layout="constrained")
    for i, n_threads in enumerate(n_threads_list):
        ax.plot(
            n_list,
            out[i],
            label=f"nt={n_threads_list[i]}",
        )
        ax.set_xlabel("n")
        ax.set_ylabel("Time (s)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.legend()

    ad.configs.set_configs("min_flops", None)

    return out, n_list, n_threads_list, fig, ax


def bench_dm(
    method,
    m=8,
    n_list=None, 
    n_threads_list=None,
    n_sims=10,
    min_flops=0,
):
    fun = {
        "dmmeq": matrix.bench_dmmeq,
        "dgemv_eq": matrix.bench_dgemv_eq,
        "dgemv_add": matrix.bench_dgemv_add,
    }[method]

    ad.configs.set_configs("min_flops", min_flops)
    if n_list is None:
        n_list = 2 ** np.arange(0, 21)
    if n_threads_list is None:
        n_threads_list = 2 ** np.arange(0, 4)
    out = np.empty((len(n_threads_list), len(n_list)))
    n_max = np.max(n_list)
    x_max = np.random.normal(0, 1, (n_max, m))
    y_max = np.random.normal(0, 1, (n_max, m))
    for i, n_threads in enumerate(n_threads_list):
        for j, n in enumerate(n_list):
            x = x_max[:n]
            y = y_max[:n]
            out[i,j] = fun(x, y, n_threads, n_sims)

    fig, ax = plt.subplots(1, 1, layout="constrained")
    for i, n_threads in enumerate(n_threads_list):
        ax.plot(
            n_list,
            out[i],
            label=f"nt={n_threads_list[i]}",
        )
        ax.set_xlabel("n")
        ax.set_ylabel("Time (s)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.legend()

    ad.configs.set_configs("min_flops", None)

    return out, n_list, n_threads_list, fig, ax


def bench_sp(
    method,
    n_list=None, 
    n_threads_list=None,
    n_sims=10,
    min_flops=0,
):
    fun = {
        "spddot": matrix.bench_spddot,
    }[method]

    ad.configs.set_configs("min_flops", min_flops)
    if n_list is None:
        n_list = 2 ** np.arange(0, 21)
    if n_threads_list is None:
        n_threads_list = 2 ** np.arange(0, 4)
    out = np.empty((len(n_threads_list), len(n_list)))
    for i, n_threads in enumerate(n_threads_list):
        for j, n in enumerate(n_list):
            inner = np.sort(np.random.choice(n, n // 2, replace=False))
            value = np.random.normal(0, 1, inner.shape[0])
            x = np.random.normal(0, 1, n)
            out[i,j] = fun(inner, value, x, n_threads, n_sims)

    fig, ax = plt.subplots(1, 1, layout="constrained")
    for i, n_threads in enumerate(n_threads_list):
        ax.plot(
            n_list,
            out[i],
            label=f"nt={n_threads_list[i]}",
        )
        ax.set_xlabel("n")
        ax.set_ylabel("Time (s)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.legend()

    ad.configs.set_configs("min_flops", None)

    return out, n_list, n_threads_list, fig, ax