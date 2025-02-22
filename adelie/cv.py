from typing import Union
from .diagnostic import (
    coefficient,
    predict,
)
from .glm import (
    GlmBase32,
    GlmBase64,
    GlmMultiBase32,
    GlmMultiBase64,
)
from .matrix import (
    MatrixNaiveBase32,
    MatrixNaiveBase64,
)
from .solver import grpnet
from . import logger
from . import matrix
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse


@dataclass
class CVGrpnetResult:
    """Result of K-fold CV group elastic net.
    """

    lmdas: np.ndarray
    """
    The common regularization path used for all folds.
    """
    losses: np.ndarray
    """
    ``losses[k,i]`` is the CV loss when validating on fold ``k`` at ``lmdas[i]``.
    """
    avg_losses: np.ndarray
    """
    ``avg_losses[i]`` is the average CV loss at ``lmdas[i]``.
    """
    best_idx: int
    """
    Argmin of ``avg_losses``.
    """
    
    def plot_loss(self):
        """Plots the average K-fold CV loss.

        For each fitted :math:`\\lambda`, the average K-fold CV loss
        as well as an error bar of one standard deviation (above and below) is plotted.
        """
        ts = -np.log(self.lmdas)
        avg_losses = np.mean(self.losses, axis=0)
        std_losses = np.std(self.losses, axis=0, ddof=0)

        fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")

        ax.errorbar(
            x=ts,
            y=avg_losses,
            yerr=std_losses,
            linestyle="None",
            marker=".",
            ecolor="grey",
            elinewidth=0.5,
            color="red",
            capsize=2,
        )
        ax.set_title("K-Fold CV Mean Loss")
        ax.set_xlabel(r"$-\log(\lambda)$")
        ax.set_ylabel("Mean Loss")

        return fig, ax

    def fit(
        self, 
        X: Union[np.ndarray, MatrixNaiveBase32, MatrixNaiveBase64],
        glm: Union[GlmBase32, GlmBase64, GlmMultiBase32, GlmMultiBase64],
        **grpnet_params,
    ):
        """Fits group elastic net until the best CV :math:`\\lambda`.

        Parameters
        ----------
        X : (n, p) Union[ndarray, MatrixNaiveBase32, MatrixNaiveBase64]
            Feature matrix.
            It is typically one of the matrices defined in :mod:`adelie.matrix` submodule or :class:`numpy.ndarray`.
        glm : Union[GlmBase32, GlmBase64, GlmMultiBase32, GlmMultiBase64]
            GLM object.
            It is typically one of the GLM classes defined in :mod:`adelie.glm` submodule.
        **grpnet_params : optional
            Parameters to :func:`adelie.solver.grpnet`.

        Returns
        -------
        state
            Result of calling :func:`adelie.solver.grpnet`.

        See Also
        --------
        adelie.solver.grpnet
        """
        logger_level = logger.logger.level
        logger.logger.setLevel(logger.logging.ERROR)
        state = grpnet(
            X=X,
            glm=glm,
            lmda_path_size=0,
            progress_bar=False,
        )
        logger.logger.setLevel(logger_level)

        lmda_path_size = 100
        if "lmda_path_size" in grpnet_params:
            lmda_path_size = grpnet_params["lmda_path_size"]
        lmda_star = self.lmdas[self.best_idx]
        full_lmdas = state.lmda_max * np.logspace(
            0, np.log10(lmda_star / state.lmda_max), lmda_path_size
        )
        return grpnet(
            X=X,
            glm=glm,
            lmda_path=full_lmdas,
            early_exit=False,
            **grpnet_params,
        )


def cv_grpnet(
    X: Union[np.ndarray, MatrixNaiveBase32, MatrixNaiveBase64],
    glm: Union[GlmBase32, GlmBase64, GlmMultiBase32, GlmMultiBase64],
    *,
    n_threads: int =1,
    early_exit: bool =False,
    min_ratio: float =1e-1,
    lmda_path_size: int =100,
    n_folds: int =5,
    seed: int =None,
    **grpnet_params,
):
    """Solves cross-validated group elastic net via naive method.

    This function was written with the intent that ``glm``
    is to be one of the GLMs defined in :mod:`adelie.glm`.
    In particular, we assume the observation weights ``w`` associated with ``glm``
    has the property that if ``w[i] == 0``,
    then the ``i`` th prediction :math:`\\eta_i` is ignored in the computation of the loss.

    Parameters
    ----------
    X : (n, p) Union[ndarray, MatrixNaiveBase32, MatrixNaiveBase64]
        Feature matrix.
        It is typically one of the matrices defined in :mod:`adelie.matrix` submodule or :class:`numpy.ndarray`.
    glm : Union[GlmBase32, GlmBase64, GlmMultiBase32, GlmMultiBase64]
        GLM object.
        It is typically one of the GLM classes defined in :mod:`adelie.glm` submodule.
    n_threads : int, optional
        Number of threads.
        Default is ``1``.
    early_exit : bool, optional
        ``True`` if the function should early exit based on training deviance explained.
        Unlike in :func:`adelie.solver.grpnet`, the default value is ``False``.
        This is because internally, we construct a *common* regularization path that
        roughly contains every generated path using each training fold.
        If ``early_exit`` is ``True``, then some training folds may not fit some smaller :math:`\\lambda`'s,
        in which case, an extrapolation method is used based on :func:`adelie.diagnostic.coefficient`.
        To avoid misinterpretation of the CV loss curve for the general user,
        we disable early exiting and fit on the entire (common) path for every training fold.
        If ``early_exit`` is ``True``, the user may see a flat component to the *right* of the loss curve.
        The user must be aware that this may then be due to the extrapolation giving the same coefficients.
        Default is ``False``.
    min_ratio : float, optional
        The ratio between the largest and smallest :math:`\\lambda` in the regularization sequence.
        Unlike in :func:`adelie.solver.grpnet`, the default value is *increased*.
        This is because CV tends to pick a :math:`\\lambda` early in the path.
        If the loss curve does not look bowl-shaped, the user may decrease this value
        to fit further down the regularization path.
        Default is ``1e-1``.
    lmda_path_size : int, optional
        Number of regularizations in the path.
        Default is ``100``.
    n_folds : int, optional
        Number of CV folds.
        Default is ``5``.
    seed : int, optional
        Seed for random number generation.
        If ``None``, the seed is not explicitly set.
        Default is ``None``.
    **grpnet_params : optional
        Parameters to :func:`adelie.solver.grpnet`.
        The following cannot be specified:

            - ``ddev_tol``: internally enforced to be ``0``.
              Otherwise, the solver may stop too early when ``early_exit=True``. 

    Returns
    -------
    result : CVGrpnetResult
        Result of running K-fold CV.

    See Also
    --------
    adelie.cv.CVGrpnetResult
    adelie.solver.grpnet
    """
    X_raw = X

    if isinstance(X, np.ndarray):
        X = matrix.dense(X, method="naive", n_threads=n_threads)

    assert (
        isinstance(X, matrix.MatrixNaiveBase64) or
        isinstance(X, matrix.MatrixNaiveBase32)
    )

    n = X.rows()

    if not (seed is None):
        np.random.seed(seed)
    order = np.random.choice(n, n, replace=False)

    fold_size = n // n_folds
    remaining = n % n_folds

    # full data lambda sequence
    logger_level = logger.logger.level
    logger.logger.setLevel(logger.logging.ERROR)
    state = grpnet(
        X=X_raw,
        glm=glm,
        n_threads=n_threads,
        lmda_path_size=0,
        progress_bar=False,
    )
    full_lmdas = state.lmda_max * np.logspace(0, np.log10(min_ratio), lmda_path_size)

    cv_losses = np.empty((n_folds, full_lmdas.shape[0]))
    for fold in range(n_folds):
        # current validation fold range
        begin = (
            (fold_size + 1) * min(fold, remaining) + 
            max(fold - remaining, 0) * fold_size
        )
        curr_fold_size = fold_size + (fold < remaining)

        # mask out validation fold
        weights = glm.weights.copy()
        weights[order[begin:begin+curr_fold_size]] = 0
        weights_sum = np.sum(weights)
        weights /= weights_sum
        glm_c = glm.reweight(weights)

        # initial call to compute current lambda path augmented with full path
        state = grpnet(
            X=X_raw,
            glm=glm_c,
            n_threads=n_threads,
            lmda_path_size=0,
            progress_bar=False,
        )
        curr_lmdas = state.lmda_max * np.logspace(0, np.log10(min_ratio), lmda_path_size)
        curr_lmdas = curr_lmdas[curr_lmdas > full_lmdas[0]]
        aug_lmdas = np.sort(np.concatenate([full_lmdas, curr_lmdas]))[::-1]

        # fit on training fold
        state = grpnet(
            X=X_raw,
            glm=glm_c,
            ddev_tol=0,
            n_threads=n_threads,
            early_exit=early_exit,
            lmda_path=aug_lmdas,
            **grpnet_params,
        )

        # compute validation weight sum
        weights_sum_val = np.sum(glm.weights[order[begin:begin+curr_fold_size]])

        # get coefficients/intercepts only on full_lmdas
        betas = state.betas
        intercepts = state.intercepts
        lmdas = state.lmdas
        beta_ints = [
            coefficient(
                lmda=lmda,
                betas=betas,
                intercepts=intercepts,
                lmdas=lmdas,
            )
            for lmda in full_lmdas
        ]
        full_betas = scipy.sparse.vstack([x[0] for x in beta_ints])
        full_intercepts = np.array([x[1] for x in beta_ints])

        # compute linear predictions
        etas = predict(
            X=X_raw,
            betas=full_betas,
            intercepts=full_intercepts,
            offsets=state._offsets,
            n_threads=n_threads,
        )

        # compute loss on full data
        full_data_losses = np.array([glm.loss(eta) for eta in etas])
        # compute loss on training data
        train_losses = weights_sum * np.array([glm_c.loss(eta) for eta in etas])
        # compute induced loss on validation data
        cv_losses[fold] = (
            (full_data_losses - train_losses) / weights_sum_val
            if weights_sum_val > 0 else
            0
        )
    logger.logger.setLevel(logger_level)

    avg_losses = np.mean(cv_losses, axis=0)
    best_idx = np.argmin(avg_losses)

    return CVGrpnetResult(
        lmdas=full_lmdas,
        losses=cv_losses,
        avg_losses=avg_losses,
        best_idx=best_idx,
    )
