from adelie.adelie_core import matrix
import adelie as ad
import matplotlib.pyplot as plt
import numpy as np


def bench_dvaddi(
    n_list=None, 
    n_threads_list=None,
    n_sims=10,
):
    ad.configs.set_configs("min_flops_per_thread", 0)

    if n_list is None:
        n_list = 2 ** np.arange(0, 24)
    if n_threads_list is None:
        n_threads_list = 2 ** np.arange(0, 4)
    out = np.empty((len(n_threads_list), len(n_list)))
    for j, n in enumerate(n_list):
        x = np.random.normal(0, 1, n)
        y = np.random.normal(0, 1, n)
        for i, n_threads in enumerate(n_threads_list):
            out[i,j] = matrix.bench_dvaddi(x, y, n_threads, n_sims)

    flops_per_thread_list = n_list[None] / n_threads_list[:, None]

    fig, axes = plt.subplots(1, 2, layout="constrained")
    for i, n_threads in enumerate(n_threads_list):
        axes[0].plot(
            n_list,
            out[i],
            label=f"nt={n_threads_list[i]}",
        )
        axes[0].set_xlabel("n")
        axes[0].set_ylabel("Time (s)")
        axes[0].set_xscale("log", base=2)
        axes[0].set_yscale("log", base=2)
        axes[0].legend()

        if n_threads == 1: continue
        axes[1].plot(
            flops_per_thread_list[i],
            out[i], 
            label=f"nt={n_threads_list[i]}",
        )
        axes[1].set_xlabel("n / number of threads (n / nt)")
        axes[1].set_ylabel("Time (s)")
        axes[1].set_xscale("log", base=2)
        axes[1].set_yscale("log")
        axes[1].legend()

    ad.configs.set_configs("min_flops_per_thread")

    return out, n_list, n_threads_list, fig, axes