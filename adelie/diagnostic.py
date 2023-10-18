from . import logger
import itertools
import numpy as np
import matplotlib.pyplot as plt


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
