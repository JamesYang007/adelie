from typing import Union
from .solver import grpnet
from .diagnostic import (
    coefficient,
    predict,
)
from . import glm
from . import logger
from . import matrix
from dataclasses import dataclass
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt


@dataclass
class CVGrpnetResult:
    """Result of K-fold CV group elastic net.
    """
    lmdas: np.ndarray
    losses: np.ndarray
    avg_losses: np.ndarray
    best_idx: int
    
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
        *, 
        lmda_path_size: int =100,
        **grpnet_params,
    ):
        """Fit group elastic net until the best CV :math:`\\lambda`.

        Parameters
        ----------
        lmda_path_size : int, optional
            Number of regularizations in the path.
            Default is ``100``.
        **grpnet_params
            Parameters to ``adelie.solver.grpnet``.

        Returns
        -------
        state
            Result of calling ``adelie.solver.grpnet``.

        See Also
        --------
        adelie.solver.grpnet
        """
        logger_level = logger.logger.level
        logger.logger.setLevel(logger.logging.ERROR)
        state = grpnet(
            X=grpnet_params["X"],
            glm=grpnet_params["glm"],
            lmda_path_size=0,
            progress_bar=False,
        )
        logger.logger.setLevel(logger_level)

        lmda_star = self.lmdas[self.best_idx]
        full_lmdas = state.lmda_max * np.logspace(
            0, np.log10(lmda_star / state.lmda_max), lmda_path_size
        )
        return grpnet(
            lmda_path=full_lmdas,
            early_exit=False,
            **grpnet_params,
        )


def cv_grpnet(
    *,
    X: np.ndarray,
    glm: Union[glm.GlmBase32, glm.GlmBase64],
    n_threads: int =1,
    early_exit: bool =False,
    min_ratio: float =1e-1,
    lmda_path_size: int =100,
    n_folds: int =5,
    seed: int =None,
    **grpnet_params,
):
    """Cross-validation group elastic net solver.

    Parameters
    ----------
    X : (n, p) matrix-like
        Feature matrix.
        It is typically one of the matrices defined in ``adelie.matrix`` submodule or ``np.ndarray``.
    glm : Union[adelie.glm.GlmBase32, adelie.glm.GlmBase64, adelie.glm.GlmMultiBase32, adelie.glm.GlmMultiBase64]
        GLM object.
        It is typically one of the GLM classes defined in ``adelie.glm`` submodule.
    n_threads : int, optional
        Number of threads.
        Default is ``1``.
    early_exit : bool, optional
        ``True`` if the function should early exit based on training deviance explained.
        Unlike in ``adelie.solver.grpnet``, the default value is ``False``.
        This is because internally, we construct a *common* regularization path that
        roughly contains every generated path using each training fold.
        If ``early_exit`` is ``True``, then some training folds may not fit some smaller :math:`\\lambda`'s,
        in which case, an extrapolation method is used based on ``adelie.diagnostic.coefficient``.
        To avoid misinterpretation of the CV loss curve for the general user,
        we disable early exiting and fit on the entire (common) path for every training fold.
        If ``early_exit`` is ``True``, the user may see a flat component to the *right* of the loss curve.
        The user must be aware that this may then be due to the extrapolation giving the same coefficients.
        Default is ``False``.
    min_ratio : float, optional
        The ratio between the largest and smallest :math:`\\lambda` in the regularization sequence.
        Unlike in ``adelie.solver.grpnet``, the default value is *increased*.
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
    **grpnet_params
        Parameters to ``adelie.solver.grpnet``.
        The following cannot be specified:

            - ``ddev_tol``: internally enforced to be ``0``.
              Otherwise, the solver may stop too early when ``early_exit=True``. 

    Returns
    -------
    result : CVGrpnetResult
        Result of running K-fold CV.

    See Also
    --------
    adelie.solver.grpnet
    adelie.cv.CVGrpnetResult
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

    # augmented lambda sequence
    glms = []
    lmdas = []
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
        glms.append(glm_c)

        state = grpnet(
            X=X_raw,
            glm=glm_c,
            n_threads=n_threads,
            lmda_path_size=0,
            progress_bar=False,
        )
        curr_lmdas = state.lmda_max * np.logspace(0, np.log10(min_ratio), lmda_path_size)
        curr_lmdas = curr_lmdas[
            (curr_lmdas > full_lmdas[0]) | (curr_lmdas < full_lmdas[-1])
        ]
        lmdas.append(curr_lmdas)
    aug_lmdas = np.sort(np.concatenate([full_lmdas] + lmdas))[::-1]

    # fit each training fold on the common augmented path
    cv_losses = np.empty((n_folds, aug_lmdas.shape[0]))
    for fold, glm_fold in enumerate(glms):
        # current validation fold range
        begin = (
            (fold_size + 1) * min(fold, remaining) + 
            max(fold - remaining, 0) * fold_size
        )
        curr_fold_size = fold_size + (fold < remaining)

        state = grpnet(
            X=X_raw,
            glm=glm_fold,
            ddev_tol=0,
            n_threads=n_threads,
            early_exit=early_exit,
            lmda_path=aug_lmdas,
            **grpnet_params,
        )

        weights = np.zeros(n)
        weights[order[begin:begin+curr_fold_size]] = glm.weights[order[begin:begin+curr_fold_size]]
        weights_sum = np.sum(weights)
        weights /= weights_sum
        glm_c = glm_fold.reweight(weights)

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
            for lmda in aug_lmdas
        ]
        aug_betas = scipy.sparse.vstack([x[0] for x in beta_ints])
        aug_intercepts = np.array([x[1] for x in beta_ints])
        etas = predict(
            X=X_raw,
            betas=aug_betas,
            intercepts=aug_intercepts,
            offsets=state._offsets,
            n_threads=n_threads,
        )
        cv_losses[fold] = np.array([glm_c.loss(eta) for eta in etas])
    logger.logger.setLevel(logger_level)

    avg_losses = np.mean(cv_losses, axis=0)
    best_idx = np.argmin(avg_losses)

    return CVGrpnetResult(
        lmdas=aug_lmdas,
        losses=cv_losses,
        avg_losses=avg_losses,
        best_idx=best_idx,
    )
