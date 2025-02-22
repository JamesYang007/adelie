from adelie.adelie_core.matrix import utils
from time import time
import adelie as ad
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy


def bench_dv(
    method,
    n_list=None, 
    n_threads_list=None,
    n_sims=10,
    min_bytes=0,
):
    fun = {
        "dvaddi": utils.bench_dvaddi,
        "dvsubi": utils.bench_dvsubi,
        "dvveq": utils.bench_dvveq,
        "dvzero": utils.bench_dvzero,
        "ddot": utils.bench_ddot,
        "ddot_opt": utils.bench_ddot_opt,
    }[method]

    ad.configs.set_configs("min_bytes", min_bytes)
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

    ad.configs.set_configs("min_bytes", None)

    return out, n_list, n_threads_list, fig, ax


def bench_dm(
    method,
    m=8,
    n_list=None, 
    n_threads_list=None,
    n_sims=10,
    min_bytes=0,
):
    fun = {
        "dmmeq": utils.bench_dmmeq,
        "dgemv_eq": utils.bench_dgemv_eq,
        "dgemv_add": utils.bench_dgemv_add,
    }[method]

    ad.configs.set_configs("min_bytes", min_bytes)
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

    ad.configs.set_configs("min_bytes", None)

    return out, n_list, n_threads_list, fig, ax


def bench_sp(
    method,
    n_list=None, 
    n_threads_list=None,
    n_sims=10,
    min_bytes=0,
):
    fun = {
        "spddot": utils.bench_spddot,
        "spaxi": utils.bench_spaxi,
    }[method]

    ad.configs.set_configs("min_bytes", min_bytes)
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

    ad.configs.set_configs("min_bytes", None)

    return out, n_list, n_threads_list, fig, ax


def bench_naive(X, q=8, L=10, skip_cov=False):
    n, p = X.rows(), X.cols()
    df = pd.DataFrame()

    weights = np.ones(n)
    sqrt_weights = np.sqrt(weights)

    v = np.random.normal(0, 1, n)
    start = time()
    X.cmul(0, v, weights)
    elapsed = time() - start
    df["cmul"] = [elapsed * 1e3]

    v = 1.3
    out = np.empty(n)
    start = time()
    X.ctmul(0, v, out)
    elapsed = time() - start
    df["ctmul"] = [elapsed * 1e3]

    v = np.random.normal(0, 1, n)
    out = np.empty(q)
    start = time()
    X.bmul(0, q, v, weights, out)
    elapsed = time() - start
    df["bmul"] = [elapsed * 1e3]

    v = np.random.normal(0, 1, q)
    out = np.empty(n)
    start = time()
    X.btmul(0, q, v, out)
    elapsed = time() - start
    df["btmul"] = [elapsed * 1e3]

    v = np.random.normal(0, 1, n)
    out = np.empty(p)
    start = time()
    X.mul(v, weights, out)
    elapsed = time() - start
    df["mul"] = [elapsed * 1e3]

    if not skip_cov:
        v = np.random.normal(0, 1, n)
        out = np.empty((q, q), order="F")
        start = time()
        X.cov(0, q, sqrt_weights, out)
        elapsed = time() - start
        df["cov"] = [elapsed * 1e3]

    v = np.random.normal(0, 1, (L, p))
    v.ravel()[np.random.binomial(1, 0.5, v.size)] = 0
    v = scipy.sparse.csr_matrix(v)
    out = np.empty((L, n))
    start = time()
    X.sp_tmul(v, out)
    elapsed = time() - start
    df["sp_tmul"] = [elapsed * 1e3]

    return df


def bench_io(
    method,
    n_list=None, 
    n_threads_list=None,
    n_sims=10,
    min_bytes=0,
):
    fun = {
        "snp_unphased_dot": utils.bench_snp_unphased_dot,
        "snp_unphased_axi": utils.bench_snp_unphased_axi,
        "snp_phased_ancestry_dot": utils.bench_snp_phased_ancestry_dot,
        "snp_phased_ancestry_axi": utils.bench_snp_phased_ancestry_axi,
    }[method]

    ad.configs.set_configs("min_bytes", min_bytes)
    if n_list is None:
        n_list = 2 ** np.arange(0, 21)
    if n_threads_list is None:
        n_threads_list = 2 ** np.arange(0, 4)
    out = np.empty((len(n_threads_list), len(n_list)))
    for i, n_threads in enumerate(n_threads_list):
        for j, n in enumerate(n_list):
            filename = "/tmp/bench_snp_tmp.snpdat"
            if "_unphased_" in method:
                io = ad.io.snp_unphased(filename)
                data = ad.data.snp_unphased(n, 1, one_ratio=0.35, missing_ratio=0.1, two_ratio=0.05)
                io.write(data["X"])
                io.read()
            elif "_phased_" in method:
                io = ad.io.snp_phased_ancestry(filename)
                A = 8
                data = ad.data.snp_phased_ancestry(n, 1, A, one_ratio=0.45, two_ratio=0.05)
                io.write(data["X"], data["ancestries"], A)
                io.read()
            v = np.ones(n)
            out[i,j] = fun(io, 0, v, n_threads, n_sims)
            os.remove(filename)

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

    ad.configs.set_configs("min_bytes", None)

    return out, n_list, n_threads_list, fig, ax


def bench_naive(X, q=8, L=10, skip_cov=False):
    n, p = X.rows(), X.cols()
    df = pd.DataFrame()

    weights = np.ones(n)
    sqrt_weights = np.sqrt(weights)

    v = np.random.normal(0, 1, n)
    start = time()
    X.cmul(0, v, weights)
    elapsed = time() - start
    df["cmul"] = [elapsed * 1e3]

    v = 1.3
    out = np.empty(n)
    start = time()
    X.ctmul(0, v, out)
    elapsed = time() - start
    df["ctmul"] = [elapsed * 1e3]

    v = np.random.normal(0, 1, n)
    out = np.empty(q)
    start = time()
    X.bmul(0, q, v, weights, out)
    elapsed = time() - start
    df["bmul"] = [elapsed * 1e3]

    v = np.random.normal(0, 1, q)
    out = np.empty(n)
    start = time()
    X.btmul(0, q, v, out)
    elapsed = time() - start
    df["btmul"] = [elapsed * 1e3]

    v = np.random.normal(0, 1, n)
    out = np.empty(p)
    start = time()
    X.mul(v, weights, out)
    elapsed = time() - start
    df["mul"] = [elapsed * 1e3]

    if not skip_cov:
        v = np.random.normal(0, 1, n)
        out = np.empty((q, q), order="F")
        start = time()
        X.cov(0, q, sqrt_weights, out)
        elapsed = time() - start
        df["cov"] = [elapsed * 1e3]

    v = np.random.normal(0, 1, (L, p))
    v.ravel()[np.random.binomial(1, 0.5, v.size)] = 0
    v = scipy.sparse.csr_matrix(v)
    out = np.empty((L, n))
    start = time()
    X.sp_tmul(v, out)
    elapsed = time() - start
    df["sp_tmul"] = [elapsed * 1e3]

    return df