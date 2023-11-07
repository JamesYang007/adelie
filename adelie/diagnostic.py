from . import logger
import adelie as ad
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def residuals(
    state,
    *,
    y: np.ndarray,
):
    """Computes the residuals for each saved solution.

    Parameters
    ----------
    state
        A state object from solving group elastic net.
    y : (n,) np.ndarray
        Response vector.

    Returns
    -------
    resids : (l, n) np.ndarray
        Residual at each saved solution.
    """
    X = state.X
    n, p = X.rows(), X.cols()
    betas = state.betas
    intercepts = state.intercepts
    WXbs = np.empty((betas.shape[0], n))
    X.sp_btmul(0, p, betas, state.weights, WXbs)
    resids = (state.weights * y)[None] - WXbs - (state.weights[None] * intercepts[:, None])
    return resids


def gradients(
    state, 
    *, 
    resids: np.ndarray,
):
    """Computes the set of gradients for each saved solution.

    Parameters
    ----------
    state
        A state object from solving group elastic net.
    resids : (l, n) np.ndarray
        Residuals for each saved solution.

    Returns
    -------
    grads : (l, p) np.ndarray
        Gradient at each saved solution.
    """
    X = state.X
    p = X.cols()
    grads = np.empty((resids.shape[0], p))
    for i in range(resids.shape[0]):
        X.mul(resids[i], grads[i])
    return grads


def gradient_norms(
    state, 
    *, 
    grads: np.ndarray,
):
    """Computes the group-wise gradient norms for each saved solution.

    Parameters
    ----------
    state
        A state object from solving group elastic net.
    grads : (l, p) np.ndarray
        Gradients for each :math:`\\lambda` value.

    Returns
    -------
    norms : (l, G) np.ndarray
        Gradient norms at each saved solution.
    """
    return np.array([
        np.linalg.norm(grads[:, g:g+gs], axis=-1)
        for g, gs in zip(state.groups, state.group_sizes)
    ]).T


def gradient_scores(
    state, 
    *, 
    abs_grads: np.ndarray,
):
    """Computes the gradient scores for each saved solution.

    Parameters
    ----------
    state
        A state object from solving group elastic net.
    abs_grads : (l, G) np.ndarray
        Gradient norms for each :math:`\\lambda` value.

    Returns
    -------
    scores : (l, G) np.ndarray
        Gradient scores at each saved solution.  
    """
    denom = state.alpha * state.penalty
    scores = np.divide(abs_grads, denom, where=denom[None] > 0)
    scores[:, denom <= 0] = state.lmdas[:, None]
    return scores


def coefficient(
    state,
    *,
    lmda: float,
):
    """Computes the coefficient at :math:`\\lambda` using linear interpolation of solutions.

    Parameters
    ----------
    state
        A state object from solving group elastic net.
    lmda : float
        New regularization parameter at which to find the solution.

    Returns
    -------
    beta : scipy.sparse.csr_matrix
        Coefficient vector at :math:`\\lambda`.
    """
    lmdas = state.lmdas
    betas = state.betas 

    order = np.argsort(lmdas)
    idx = np.searchsorted(
        lmdas,
        lmda,
        sorter=order,
    )
    idx = lmdas.shape[0] - idx
    if idx == 0 or idx == lmdas.shape[0]:
        logger.logger.warning(
            "lmda is not within the range of the saved lambdas. " +
            "Returning boundary solution."
        )
        idx = np.clip(idx, 0, lmdas.shape[0]-1)
        return betas[idx]

    left, right = betas[idx-1], betas[idx]
    weight = (lmda - lmdas[idx]) / (lmdas[idx-1] - lmdas[idx])

    return left.multiply(weight) + right.multiply(1-weight)


def plot_coefficients(state):
    """Plots the coefficient profile.

    Parameters
    ----------
    state
        A state object from solving group elastic net.
    """
    groups = state.groups
    group_sizes = state.group_sizes
    betas = state.betas
    lmdas = state.lmdas

    tls = -np.log(lmdas)

    fig, ax = plt.subplots(layout="constrained")

    for g, gs in zip(groups, group_sizes):
        curr_block = betas[:, g:g+gs]
        if curr_block.nnz == 0:
            continue
        curr_block = curr_block.toarray()
        ax.plot(tls, curr_block, linestyle="-")

    ax.set_title("Coefficient Profile")
    ax.set_ylabel(r"$\beta$")
    ax.set_xlabel(r"-$\log(\lambda)$")

    return fig, ax


def plot_rsqs(state):
    """Plots the :math:`R^2` profile.

    Parameters
    ----------
    state
        A state object from solving group elastic net.
    """
    rsqs = state.rsqs
    lmdas = state.lmdas

    tls = -np.log(lmdas)

    fig, ax = plt.subplots(layout="constrained")
    ax.plot(tls, rsqs, linestyle='-', color='r', marker='.')
    ax.set_title(r"$R^2$ Profile")
    ax.set_ylabel(r"$R^2$")
    ax.set_xlabel(r"$-\log(\lambda)$")

    return fig, ax


def plot_set_sizes(
    state,
    *,
    ratio: bool =False,
    exclude: list =[],
    axes = None,
):
    """Plots the set sizes.

    Parameters
    ----------
    state
        A state object from solving group elastic net.
    ratio : bool, optional
        ``True`` if plot should normalize the set sizes
        by the total number of groups.
        Default is ``False``.
    exclude : list, optional
        The sets to exclude from plotting.
        It must be a subset of the following:

            - ``"active"``: active set.
            - ``"screen"``: screen set.

        Default is ``[]``.
    axes
        Matplotlib axes object.
    """
    make_ax = axes is None
    ax = axes

    include = ["active", "screen"]
    if len(exclude) > 0:
        include = list(set(include) - set(exclude))
    
    include_map = {
        "active": 0,
        "screen": 1,
    } 

    ys = [
        state.active_sizes,
        state.screen_sizes,
    ]
    if ratio:
        ys = [y / len(state.groups) for y in ys]

    labels = [
        "active",
        state.screen_rule,
    ]
    colors = [
        "tab:red",
        "tab:blue",
    ]
    markers = ["o", "v"]

    y_sizes = np.array([y.shape[0] for y in ys])
    iters = np.min(y_sizes)
    if not np.all(y_sizes == iters):
        logger.logger.warning(
            "The sets do not all have the same set sizes. " +
            "The plot will only show up to the smallest set."
        )
    tls = -np.log(state.lmdas[:iters])
    ys = [y[:iters] for y in ys]

    ys = [ys[include_map[s]] for s in include]
    labels = [labels[include_map[s]] for s in include]
    colors = [colors[include_map[s]] for s in include]
    markers = [markers[include_map[s]] for s in include]

    if make_ax:
        fig, ax = plt.subplots(layout="constrained")

    for y, marker, color, label in zip(ys, markers, colors, labels):
        ax.plot(
            tls,
            y, 
            linestyle="None", 
            marker=marker,
            markerfacecolor="None",
            color=color,
            label=label,
        )
    ax.legend()
    ax.set_title("Set Size Profile")
    if ratio:
        ax.set_ylabel("Proportion of Groups")
    else:
        ax.set_ylabel("Number of Groups")
    ax.set_xlabel(r"$-\log(\lambda)$")

    if make_ax:
        return fig, ax
    return ax


def plot_benchmark(state):
    """Plots benchmark times.

    Parameters
    ----------
    state
        A state object from solving group elastic net.
    """
    times = [
        state.benchmark_screen,
        state.benchmark_fit_screen,
        state.benchmark_fit_active,
        state.benchmark_kkt,
        state.benchmark_invariance,
    ]
    n_iters = np.min([len(t) for t in times])
    times = [t[:n_iters] for t in times]

    colors = [
        "green",
        "orange",
        "red",
        "purple",
        "brown",
    ]
    markers = [
        ".", "v", "^", "+", "*",
    ]
    labels = [
        "screen", "fit-strong", "fit-active", "kkt", "invariance",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9, 6), layout="constrained")
    xs = np.arange(n_iters)
    for tm, color, marker, label in zip(times, colors, markers, labels):
        axes[0].plot(
            xs,
            tm,
            linestyle="None",
            color=color,
            marker=marker,
            markerfacecolor="None",
            label=label,
        )
    axes[0].legend()
    axes[0].set_title("Benchmark Profile")
    axes[0].set_ylabel("Time (s)")
    axes[0].set_xlabel("BASIL Iteration")
    axes[0].set_yscale("log")

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
    state,
    *,
    scores: np.ndarray, 
    idx: int =None,
):
    """Plots KKT failures.

    Parameters
    ----------
    state
        A state object from solving group elastic net.
    scores : (p,) np.ndarray
        Gradient vector.
    """
    G = scores.shape[-1]
    lmdas = state.lmdas

    scores = scores / state.lmdas[:, None] - 1

    do_anim = idx is None
    idx = 0 if do_anim else idx

    gns = np.arange(G)

    colors = ["blue", "red"]
    labels = ["success", "failure"]
    alphas = [0.6, 0.8]

    fig, ax = plt.subplots(layout="constrained")

    is_failure = scores[idx] > 0
    xs = [
        gns[~is_failure],
        gns[is_failure],
    ]
    ys = [
        scores[idx, ~is_failure],
        scores[idx, is_failure],
    ]
    scats = [None] * 2
    for i, (x, y, color, label, alpha) in enumerate(zip(xs, ys, colors, labels, alphas)):
        scats[i] = ax.scatter(
            x, y,
            color=color,
            marker='.',
            facecolor="None",
            label=label,
            alpha=alpha,
        )
    ax.legend()
    bound = np.maximum(np.max(scores[idx]) * 1.05, 1e-5)
    ax.set_ylim(
        bottom=-bound,
        top=bound,
    )
    ax.axhline(0, linestyle='--', linewidth=1, color="green")
    ax.set_title("Scaled Active Score Error (Largest)")
    ax.set_ylabel("Scaled Active Score")
    ax.set_xlabel("Group Number")


    if do_anim:
        plt.close(fig)
    else:
        return fig, ax

    def update(idx):
        s = scores[idx]

        is_failure = s > 0
        xs = [
            gns[~is_failure],
            gns[is_failure],
        ]
        ys = [
            s[~is_failure],
            s[is_failure],
        ]
        for i, (x, y, color, label, alpha) in enumerate(zip(xs, ys, colors, labels, alphas)):
            data = np.stack([x, y]).T
            scats[i].set_offsets(data)
            scats[i].set(
                color=color,
                facecolor="None",
                alpha=alpha,
                label=label,
            )
        bound = np.maximum(np.max(s) * 1.05, 1e-5)
        ax.set_ylim(
            bottom=-bound,
            top=bound,
        )
        return (scats[0], scats[1],)

    anim = animation.FuncAnimation(
        fig=fig, 
        func=update, 
        frames=lmdas.shape[0]-1, 
        interval=200, 
        repeat=False,
    )

    return anim.to_html5_video()


class Diagnostic:
    """Diagnostic class for user-friendly API.

    Parameters
    ----------
    y : (n,) np.ndarray
        Response vector.
    state
        A state object from solving group elastic net.
    """
    def __init__(self, *, y, state):
        self.state = state
        self.residuals = residuals(state, y=y)
        self.gradients = gradients(state, resids=self.residuals)
        self.gradient_norms = gradient_norms(state, grads=self.gradients)
        self.gradient_scores = gradient_scores(state, abs_grads=self.gradient_norms)

    def plot_coefficients(self):
        """Plots the coefficient profile.

        See Also
        --------
        adelie.diagnostic.plot_coefficients
        """
        return plot_coefficients(self.state)

    def plot_rsqs(self):
        """Plots the :math:`R^2` profile.

        See Also
        --------
        adelie.diagnostic.plot_rsqs
        """
        return plot_rsqs(self.state)

    def plot_set_sizes(self, **kwargs):
        """Plots the set sizes.

        See Also
        --------
        adelie.diagnostic.plot_set_sizes
        """
        return plot_set_sizes(self.state, **kwargs)

    def plot_benchmark(self):
        """Plots benchmark times.

        See Also
        --------
        adelie.diagnostic.plot_benchmark
        """
        return plot_benchmark(self.state)

    def plot_kkt(self, **kwargs):
        """Plots KKT failures.

        See Also
        --------
        adelie.diagnostic.plot_kkt
        """
        return plot_kkt(self.state, scores=self.gradient_scores, **kwargs)
