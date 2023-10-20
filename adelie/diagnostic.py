from . import logger
import itertools
import numpy as np
import matplotlib.pyplot as plt


def active_sets(state):
    p = state.X.cols()
    feature_to_group = np.empty(p, dtype=int)
    for i, (g, gs) in enumerate(zip(state.groups, state.group_sizes)):
        feature_to_group[g:g+gs] = i
    active_sets = [
        set(feature_to_group[state.betas[i].indices])
        for i in range(state.betas.shape[0])
    ]
    return active_sets



def gradients(X, y, state):
    """Computes the set of gradients for each saved solution.

    Parameters
    ----------
    X : (n, p) np.ndarray
        Feature matrix.
    y : (n,) np.ndarray
        Response vector.
    state
        A state object from solving group elastic net.
    """
    betas = state.betas
    resids = y[None] - betas @ X.T
    if state.intercept:
        resids -= np.mean(y)
    grads = resids @ X
    if state.intercept:
        grads -= state.X_means[None] * np.sum(resids, axis=-1)[:, None]
    return grads


def gradient_norms(grads, state):
    """Computes the group-wise gradient norms.

    Parameters
    ----------
    grads : (l, p) np.ndarray
        Gradients for each :math:`\\lambda` value.
    state
        A state object from solving group elastic net.
    """
    return np.array([
        np.linalg.norm(grads[:, g:g+gs], axis=-1)
        for g, gs in zip(state.groups, state.group_sizes)
    ]).T


def plot_coefficients(
    state,
):
    """Plots the coefficient profile.

    Parameters
    ----------
    state
        A state object from solving group elastic net.
    """
    groups = state.groups
    group_sizes = state.group_sizes
    betas = state.betas
    intercepts = state.intercepts
    lmdas = state.lmdas

    tls = -np.log(lmdas)

    fig = plt.figure(layout="constrained")

    for g, gs in zip(groups, group_sizes):
        curr_block = betas[:, g:g+gs]
        if curr_block.nnz == 0:
            continue
        curr_block = curr_block.toarray()
        plt.plot(tls, curr_block, linestyle="-")

    plt.plot(tls, intercepts, linestyle='-')

    plt.title("Coefficient Profile")
    plt.ylabel(r"$\beta$")
    plt.xlabel(r"-$\log(\lambda)$")

    return fig 


def plot_rsqs(
    state,
):
    """Plots the :math:`R^2` profile.

    Parameters
    ----------
    state
        A state object from solving group elastic net.
    """
    rsqs = state.rsqs
    lmdas = state.lmdas

    tls = -np.log(lmdas)

    fig = plt.figure(layout="constrained")
    plt.plot(tls, rsqs, linestyle='-', color='r', marker='.')
    plt.title(r"$R^2$ Profile")
    plt.ylabel(r"$R^2$")
    plt.xlabel(r"$-\log(\lambda)$")

    return fig


def plot_set_sizes(
    state,
    ratio: bool =True,
):
    """Plots the active, strong, and EDPP set sizes.

    Parameters
    ----------
    state
        A state object from solving group elastic net.
    ratio : bool, optional
        ``True`` if plot should normalize the set sizes
        by the total number of groups.
        Default is ``True``.
    """
    ys = [
        state.active_sizes,
        state.strong_sizes,
        state.edpp_safe_sizes,
    ]
    if ratio:
        ys = [y / len(state.groups) for y in ys]

    names = [
        "active",
        "strong",
        "safe",
    ]

    y_sizes = np.array([y.shape[0] for y in ys])
    iters = np.min(y_sizes)
    if not np.all(y_sizes == iters):
        logger.logger.warning(
            "The sets do not all have the same set sizes. " +
            "The plot will only show up to the smallest set."
        )

    tls = -np.log(state.lmdas[:iters])
    ys = [y[:iters] for y in ys]

    marker = itertools.cycle(('o', 'v', '^', '<', '>', 's', '8', 'p'))

    fig = plt.figure(layout="constrained")

    for name, y in zip(names, ys):
        plt.plot(
            tls,
            y, 
            linestyle="None", 
            marker=next(marker),
            markerfacecolor="None",
            label=name,
        )
    plt.legend()
    plt.title("Set Size Profile")
    if ratio:
        plt.ylabel("Proportion of Groups")
    else:
        plt.ylabel("Number of Groups")
    plt.xlabel(r"$-\log(\lambda)$")

    return fig


def plot_benchmark(
    state
):
    """Plots benchmark times.

    Parameters
    ----------
    state
        A state object from solving group elastic net.
    """
    times = [
        state.benchmark_screen,
        state.benchmark_fit_strong,
        state.benchmark_fit_active,
        state.benchmark_kkt,
        state.benchmark_invariance,
    ]
    n_iters = np.min([len(t) for t in times])
    times = [t[:n_iters] for t in times]

    markers = [
        ".", "v", "^", "+", "*",
    ]
    labels = [
        "screen", "fit-strong", "fit-active", "kkt", "invariance",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9, 6), layout="constrained")
    xs = np.arange(n_iters)
    for tm, marker, label in zip(times, markers, labels):
        axes[0].plot(
            xs,
            tm,
            linestyle="None",
            marker=marker,
            markerfacecolor="None",
            label=label,
        )
    axes[0].legend()
    axes[0].set_title("Benchmark Profile")
    axes[0].set_ylabel("Time (s)")
    axes[0].set_xlabel("BASIL Iteration")

    colors = [
        "green",
        "orange",
        "red",
        "purple",
        "grey",
    ]
    total_times = np.array([np.sum(t) for t in times])
    total_times /= np.sum(total_times)
    axes[1].bar(
        np.arange(len(total_times)),
        total_times,
        color=colors,
        edgecolor=colors,
        linewidth=1.5,
        label=labels,
        alpha=0.5,
    )
    axes[1].legend()
    axes[1].set_title("Total Time")
    axes[1].set_ylabel("Proportion of Time")
    axes[1].set_xlabel("Category")

    return fig, axes


def plot_kkt(
    abs_grad, 
    lmda, 
    state,
):
    """Plots KKT failures.

    Parameters
    ----------
    grad : (p,) np.ndarray
        Gradient vector.
    lmda : float
        :math:`\\lambda` at which ``grad`` is computed.
    state
        A state object from solving group elastic net.
    """
    fig = plt.figure(layout="constrained")
    weights = abs_grad / (state.alpha * state.penalty * lmda) - 1
    colors = ["blue", "red"]
    plt.scatter(
        np.arange(len(weights)),
        weights,
        color=[colors[w > 0] for w in weights],
        marker='.',
        facecolor="None",
    )
    plt.axhline(0)
    max_weight = np.max(weights)
    plt.ylim(bottom=-max_weight * 1.05, top=max_weight * 1.05)
    plt.title("Group-wise Normalized Gradient Norms (KKT Failures)")
    plt.ylabel(r"$\|g_k\|_2 / (\alpha p_k \lambda_k) - 1$")
    plt.xlabel("Group Number")

    return fig
