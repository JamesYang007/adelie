from typing import Union
from . import logger
from .glm import (
    GlmBase32,
    GlmBase64,
    gaussian,
)
from .matrix import (
    MatrixNaiveBase32,
    MatrixNaiveBase64,
)
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def predict(
    *,
    glm: Union[GlmBase32, GlmBase64],
    X: Union[MatrixNaiveBase32, MatrixNaiveBase64],
    betas: Union[np.ndarray, scipy.sparse.csr_matrix],
    intercepts: np.ndarray,
):
    """Computes the predictions.

    The prediction is given by
    
    .. math::
        \\begin{align*}
            \\hat{y} = \\nabla \\underline{A}(X\\beta + \\beta_0 \\mathbf{1})
        \\end{align*}

    Parameters
    ----------
    glm : Union[adelie.glm.GlmBase32, adelie.glm.GlmBase64]
        GLM object.
    X : (n, p) Union[MatrixNaiveBase32, MatrixNaiveBase64]
        Feature matrix.
    betas : (l, p) Union[np.ndarray, scipy.sparse.csr_matrix]
        Matrix with each row being a coefficient vector.
    intercepts : (l,) np.ndarray
        Intercept corresponding to ``betas``.

    Returns
    -------
    preds : (l, n) np.ndarray
        Predictions.

    See Also
    --------
    adelie.glm.GlmBase64
    """
    if glm is None:
        glm = gaussian()
    n = X.rows()
    etas = np.empty((betas.shape[0], n))
    if isinstance(betas, scipy.sparse.csr_matrix):
        X.sp_btmul(betas, np.ones(n), etas)
    elif isinstance(betas, np.ndarray):
        if not betas.flags.c_contiguous:
            raise RuntimeError("betas must be C-contiguous if np.ndarray.")
        p = X.cols()
        _ones = np.ones(n)
        for i in range(betas.shape[0]):
            X.btmul(0, p, betas[i], _ones, etas[i])
    etas += intercepts[:, None]
    mus = np.empty((betas.shape[0], n))
    for i in range(betas.shape[0]):
        glm.gradient(etas[i], mus[i])
    return mus


def residuals(
    *,
    glm: Union[GlmBase32, GlmBase64],
    X: Union[MatrixNaiveBase32, MatrixNaiveBase64],
    y: np.ndarray,
    betas: Union[np.ndarray, scipy.sparse.csr_matrix],
    intercepts: np.ndarray,
):
    """Computes the residuals.

    The residual is given by 
    
    .. math::
        \\begin{align*}
            \\hat{r} = y - \\hat{y}
        \\end{align*}

    Parameters
    ----------
    glm : Union[adelie.glm.GlmBase32, adelie.glm.GlmBase64]
        GLM object.
    X : (n, p) Union[MatrixNaiveBase32, MatrixNaiveBase64]
        Feature matrix.
    y : (n,) np.ndarray
        Response vector.
    betas : (l, p) Union[np.ndarray, scipy.sparse.csr_matrix]
        Matrix with each row being a coefficient vector.
    intercepts : (l,) np.ndarray
        Intercept corresponding to ``betas``.

    Returns
    -------
    resids : (l, n) np.ndarray
        Residuals.

    See Also
    --------
    adelie.diagnostic.predict
    """
    preds = predict(
        glm=glm,
        X=X, 
        betas=betas, 
        intercepts=intercepts,
    )
    return y[None] - preds


def gradients(
    *, 
    X: Union[MatrixNaiveBase32, MatrixNaiveBase64],
    weights: np.ndarray,
    resids: np.ndarray,
):
    """Computes the gradients.

    The gradient is given by

    .. math::
        \\begin{align*}
            \\hat{\\gamma} = X^{\\top} W \\hat{r}
        \\end{align*}

    Parameters
    ----------
    X : (n, p) Union[MatrixNaiveBase32, MatrixNaiveBase64]
        Feature matrix.
    weights : (n,) np.ndarray
        Observation weights.
    resids : (l, n) np.ndarray
        Residuals.

    Returns
    -------
    grads : (l, p) np.ndarray
        Gradients.

    See Also
    --------
    adelie.diagnostic.residuals
    """
    p = X.cols()
    Wresids = resids * weights[None]
    grads = np.empty((Wresids.shape[0], p))
    for i in range(Wresids.shape[0]):
        X.mul(Wresids[i], grads[i])
    return grads


def gradient_norms(
    *, 
    groups: np.ndarray,
    group_sizes: np.ndarray,
    alpha: float,
    penalty: np.ndarray,
    lmdas: np.ndarray,
    betas: scipy.sparse.csr_matrix,
    grads: np.ndarray,
):
    """Computes the group-wise gradient norms.

    The group-wise gradient norm is given by :math:`h \\in \\mathbb{R}^{G}` where

    .. math::
        \\begin{align*}
            h_g = \\|\\hat{\\gamma}_g - \\lambda (1-\\alpha) p_g \\beta_g\\|_2  \\quad g=1,\\ldots, G
        \\end{align*}

    where
    :math:`\\hat{\\gamma}_g` is the gradient,
    :math:`p_g` is the penalty factor,
    and :math:`\\beta_g` is the coefficient block for group :math:`g`.

    Parameters
    ----------
    groups : (G,) np.ndarray
        List of starting indices to each group where `G` is the number of groups.
        ``groups[i]`` is the starting index of the ``i`` th group. 
    group_sizes : (G,) np.ndarray
        List of group sizes corresponding to each element of ``groups``.
        ``group_sizes[i]`` is the size of the ``i`` th group.
    alpha : float
        Elastic net parameter.
    penalty : (G,) np.ndarray
        Penalty factor for each group in the same order as ``groups``.
    lmdas : (l,) np.ndarray
        Regularizations.
    betas : (l, p) np.ndarray
        Coefficient vectors.
    grads : (l, p) np.ndarray
        Gradients.

    Returns
    -------
    norms : (l, G) np.ndarray
        Gradient norms.

    See Also
    --------
    adelie.diagnostic.gradients
    """
    penalty = np.repeat(penalty, group_sizes)
    grads = grads - betas.multiply(lmdas[:, None] * (1 - alpha) * penalty[None])
    return np.array([
        np.linalg.norm(grads[:, g:g+gs], axis=-1)
        for g, gs in zip(groups, group_sizes)
    ]).T


def gradient_scores(
    *, 
    alpha: float,
    penalty: np.ndarray,
    lmdas: np.ndarray,
    grad_norms: np.ndarray,
):
    """Computes the gradient scores.

    The gradient score is given by

    .. math::
        \\begin{align*}
            \\hat{s}_g = 
            \\begin{cases}
                \\hat{\\gamma}_g \\cdot (\\alpha p_g)^{-1} ,& \\alpha p_g > 0 \\\\
                \\lambda ,& \\alpha p_g = 0
            \\end{cases}
            \\qquad
            g = 1,\\ldots, G
        \\end{align*}

    Parameters
    ----------
    alpha : float
        Elastic net parameter.
    penalty : (G,) np.ndarray
        Penalty factor for each group in the same order as ``groups``.
    lmdas : (l,) np.ndarray
        Regularizations.
    grad_norms : (l, G) np.ndarray
        Gradient norms.

    Returns
    -------
    scores : (l, G) np.ndarray
        Gradient scores.  

    See Also
    --------
    adelie.diagnostic.gradient_norms
    """
    denom = alpha * penalty
    scores = np.divide(grad_norms, denom, where=denom[None] > 0)
    scores[:, denom <= 0] = lmdas[:, None]
    return scores


def coefficient(
    *,
    lmda: float,
    lmdas: np.ndarray,
    betas: scipy.sparse.csr_matrix,
):
    """Computes the coefficient at :math:`\\lambda` using linear interpolation of solutions.

    The linearly interpolated coefficient is given by
    
    .. math::
        \\begin{align*}
            \\hat{\\beta}(\\lambda)
            =
            \\frac{\\lambda - \\lambda_{k+1}}{\\lambda_{k} - \\lambda_{k+1}}
            \\hat{\\beta}(\\lambda_k)
            +
            \\frac{\\lambda_{k} - \\lambda}{\\lambda_{k} - \\lambda_{k+1}}
            \\hat{\\beta}(\\lambda_{k+1})
        \\end{align*}

    if :math:`\\lambda \\in [\\lambda_{k+1}, \\lambda_k]`.
    If :math:`\\lambda` lies above the largest value in ``lmdas`` or below the smallest value,
    then we simply take the solution at the respective ends.

    Parameters
    ----------
    lmda : float
        New regularization parameter at which to find the solution.
    lmdas : (l,) np.ndarray
        Regularizations.
    betas : (l, p) np.ndarray
        Solution coefficient vectors for each :math:`\\lambda` in ``lmdas``.

    Returns
    -------
    beta : (1, p) scipy.sparse.csr_matrix
        Linearly interpolated coefficient vector at :math:`\\lambda`.
    """
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


def plot_coefficients(
    *,
    groups: np.ndarray,
    group_sizes: np.ndarray,
    lmdas: np.ndarray,
    betas: scipy.sparse.csr_matrix,
):
    """Plots the coefficient profile.

    Parameters
    ----------
    groups : (G,) np.ndarray
        List of starting indices to each group where `G` is the number of groups.
        ``groups[i]`` is the starting index of the ``i`` th group. 
    group_sizes : (G,) np.ndarray
        List of group sizes corresponding to each element of ``groups``.
        ``group_sizes[i]`` is the size of the ``i`` th group.
    lmdas : (l,) np.ndarray
        Regularizations.
    betas : (l, p) np.ndarray
        Coefficient vectors.

    Returns
    -------
    fig, ax
    """
    tls = -np.log(lmdas)

    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")

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


def plot_devs(
    *,
    lmdas: np.ndarray,
    devs: np.ndarray,
):
    """Plots the deviance profile.

    Parameters
    ----------
    lmdas : (l,) np.ndarray
        Regularizations.
    devs : (l,) np.ndarray
        Deviances.

    Returns
    -------
    fig, ax
    """
    tls = -np.log(lmdas)

    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    ax.plot(tls, devs, linestyle='-', color='r', marker='.')
    ax.set_title(r"Deviance Profile")
    ax.set_ylabel(r"Deviance Explained (%)")
    ax.set_xlabel(r"$-\log(\lambda)$")

    return fig, ax


def plot_set_sizes(
    *,
    groups: np.ndarray,
    screen_sizes: np.ndarray,
    active_sizes: np.ndarray,
    lmdas: np.ndarray,
    screen_rule: str,
    ratio: bool =False,
    exclude: list =[],
    axes = None,
):
    """Plots the set sizes.

    Parameters
    ----------
    groups : (G,) np.ndarray
        List of starting indices to each group where `G` is the number of groups.
        ``groups[i]`` is the starting index of the ``i`` th group. 
    screen_sizes : (l,) np.ndarray
        Screen set sizes.
    active_sizes : (l,) np.ndarray
        Active set sizes.
    lmdas : (l,) np.ndarray
        Regularizations.
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

    Returns
    -------
    fig, ax
        If ``axes`` is ``None``, both are returned.
    ax
        If ``axes`` is not ``None``, then only ``ax`` is returned.
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
        active_sizes,
        screen_sizes,
    ]
    if ratio:
        ys = [y / len(groups) for y in ys]

    labels = [
        "active",
        screen_rule,
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
    tls = -np.log(lmdas[:iters])
    ys = [y[:iters] for y in ys]

    ys = [ys[include_map[s]] for s in include]
    labels = [labels[include_map[s]] for s in include]
    colors = [colors[include_map[s]] for s in include]
    markers = [markers[include_map[s]] for s in include]

    if make_ax:
        fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")

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


def plot_benchmark(
    *,
    total_time: np.ndarray,
    benchmark_screen: np.ndarray,
    benchmark_fit_screen: np.ndarray,
    benchmark_fit_active: np.ndarray,
    benchmark_kkt: np.ndarray,
    benchmark_invariance: np.ndarray,
    n_valid_solutions: np.ndarray,
    lmdas: np.ndarray,
    relative: bool =False,
):
    """Plots benchmark times.

    Parameters
    ----------
    total_time : float
        Total time taken for the core routine.
    benchmark_screen : (L,) np.ndarray
        Benchmark timings for screening.
    benchmark_fit_screen : (L,) np.ndarray
        Benchmark timings for fitting on screen set.
    benchmark_fit_active : (L,) np.ndarray
        Benchmark timings for fitting on active set.
    benchmark_kkt : (L,) np.ndarray
        Benchmark timings for KKT checks.
    benchmark_invariance : (L,) np.ndarray
        Benchmark timings for invariance step.
    n_valid_solutions : (L,) np.ndarray
        Flags that indicate whether each iteration resulted in a valid solution.
    lmdas : (l,) np.ndarray
        Regularizations.
    relative : bool, optional
        If ``True``, the time breakdown plot is relative to the total time,
        therefore plotting the proportion of time spent in each category.
        Otherwise, the absolute times are shown.
        Default is ``False``.

    Returns
    -------
    fig, ax
    """
    def _squash_times(ts):
        idx = 0
        new_ts = []
        while idx < len(n_valid_solutions):
            if n_valid_solutions[idx]:
                new_ts.append(ts[idx]) 
            else:
                t = 0
                while (idx < len(n_valid_solutions)) and (not n_valid_solutions[idx]):
                    t += ts[idx]
                    idx += 1
                if idx < len(n_valid_solutions):
                    t += ts[idx]
                new_ts.append(t)
            idx += 1
        return np.array(new_ts)

    times = [
        _squash_times(benchmark_screen),
        _squash_times(benchmark_fit_screen),
        _squash_times(benchmark_fit_active),
        _squash_times(benchmark_kkt),
        _squash_times(benchmark_invariance),
    ]
    n_iters = np.min([len(t) for t in times])
    times = [t[:n_iters] for t in times]
    lmdas = lmdas[:n_iters]
    tlmdas = -np.log(lmdas)

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
        "screen", "fit-screen", "fit-active", "kkt", "invariance",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9, 6), layout="constrained")
    for tm, color, marker, label in zip(times, colors, markers, labels):
        axes[0].plot(
            tlmdas,
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
    axes[0].set_xlabel(r"$-\log(\lambda)$")
    axes[0].set_yscale("log")

    total_times = np.array([np.sum(t) for t in times])
    total_times_sum = np.sum(total_times)
    total_times = np.concatenate([
        total_times, 
        [total_time - total_times_sum] # unaccounted time
    ])
    if relative:
        total_times /= total_time
    axes[1].bar(
        np.arange(len(total_times)),
        total_times,
        color=colors + ["grey"],
        edgecolor=colors + ["grey"],
        linewidth=1.5,
        label=labels + ["other"],
        alpha=0.5,
    )
    axes[1].legend()
    axes[1].set_title("Time Breakdown")
    if relative:
        axes[1].set_ylabel("Proportion of Time")
    else:
        axes[1].set_ylabel("Time (s)")
    axes[1].set_xlabel("Category")

    return fig, axes


def plot_kkt(
    *,
    lmdas: np.ndarray,
    scores: np.ndarray, 
    idx: int =None,
):
    """Plots KKT failures.

    Parameters
    ----------
    lmdas : (l,) np.ndarray
        Regularizations.
    scores : (p,) np.ndarray
        Gradient scores.
    idx : int, optional
        Index of ``lmdas`` and ``scores`` at which to plot the KKT failures.
        If ``None``, then an animation of the plots at every index is shown.
        Default is ``None``.
    """
    G = scores.shape[-1]

    scores = scores - lmdas[:, None]

    do_anim = idx is None
    idx = 0 if do_anim else idx

    gns = np.arange(G)

    colors = ["blue", "red"]
    labels = ["success", "failure"]
    alphas = [0.6, 0.8]

    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")

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
    bound = np.max(scores[idx]) * 1.05
    ax.set_ylim(
        bottom=-bound,
        top=bound,
    )
    ax.axhline(0, linestyle='--', linewidth=1, color="green")
    ax.set_title("Active Score Error (Largest)")
    ax.set_ylabel(r"$s_g - \lambda$")
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


class diagnostic:
    """Diagnostic class for user-friendly API.

    Parameters
    ----------
    state
        A state object from solving group elastic net.
    """
    def __init__(self, state):
        self.state = state
        self.betas = state.betas
        self.residuals = residuals(
            glm=self.state.glm,
            X=self.state.X,
            y=self.state._y, 
            betas=self.betas,
            intercepts=self.state.intercepts,
        )
        self.gradients = gradients(
            X=self.state.X,
            weights=self.state.weights,
            resids=self.residuals,
        )
        self.gradient_norms = gradient_norms(
            groups=self.state.groups,
            group_sizes=self.state.group_sizes,
            alpha=self.state.alpha,
            penalty=self.state.penalty,
            lmdas=self.state.lmdas,
            betas=self.state.betas,
            grads=self.gradients,
        )
        self.gradient_scores = gradient_scores(
            alpha=self.state.alpha,
            penalty=self.state.penalty,
            lmdas=self.state.lmdas,
            grad_norms=self.gradient_norms,
        )

    def plot_coefficients(self):
        """Plots the coefficient profile.

        See Also
        --------
        adelie.diagnostic.plot_coefficients
        """
        return plot_coefficients(
            groups=self.state.groups,
            group_sizes=self.state.group_sizes,
            lmdas=self.state.lmdas,
            betas=self.betas,
        )

    def plot_devs(self):
        """Plots the deviance profile.

        See Also
        --------
        adelie.diagnostic.plot_devs
        """
        return plot_devs(
            lmdas=self.state.lmdas,
            devs=self.state.devs,
        )

    def plot_set_sizes(self, **kwargs):
        """Plots the set sizes.

        See Also
        --------
        adelie.diagnostic.plot_set_sizes
        """
        return plot_set_sizes(
            groups=self.state.groups,
            screen_sizes=self.state.screen_sizes,
            active_sizes=self.state.active_sizes,
            lmdas=self.state.lmdas,
            screen_rule=self.state.screen_rule,
            **kwargs,
        )

    def plot_benchmark(self, **kwargs):
        """Plots benchmark times.

        See Also
        --------
        adelie.diagnostic.plot_benchmark
        """
        return plot_benchmark(
            total_time=self.state.total_time,
            benchmark_screen=self.state.benchmark_screen,
            benchmark_fit_screen=self.state.benchmark_fit_screen,
            benchmark_fit_active=self.state.benchmark_fit_active,
            benchmark_kkt=self.state.benchmark_kkt,
            benchmark_invariance=self.state.benchmark_invariance,
            n_valid_solutions=self.state.n_valid_solutions,
            lmdas=self.state.lmdas,
            **kwargs,
        )

    def plot_kkt(self, **kwargs):
        """Plots KKT failures.

        See Also
        --------
        adelie.diagnostic.plot_kkt
        """
        return plot_kkt(
            lmdas=self.state.lmdas,
            scores=self.gradient_scores,
            **kwargs,
        )
