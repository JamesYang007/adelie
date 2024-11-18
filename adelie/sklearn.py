from .adelie_core.sklearn import (
    css_cov_model_selection_fit_k_32,
    css_cov_model_selection_fit_k_64,
)
from .cv import (
    cv_grpnet, 
    CVGrpnetResult,
)
from .diagnostic import (
    predict,
)
from .glm import (
    binomial, 
    gaussian, 
    multigaussian, 
    multinomial,
    poisson, 
)
from .matrix import (
    MatrixNaiveBase32, 
    MatrixNaiveBase64,
)
from .solver import (
    grpnet,
)
from scipy.special import (
    expit, 
    softmax,
)
from sklearn.base import (
    BaseEstimator, 
    RegressorMixin,
)
from typing import (
    Any, 
    Dict, 
    Union,
)
import numpy as np
import warnings


class GroupElasticNet(BaseEstimator, RegressorMixin):
    """
    Group Elastic Net estimator with scikit-learn compatible API.

    Parameters
    ----------
    solver : str, optional
        The solver to use. 
        It must be one of the following:

            - ``"grpnet"``
            - ``"cv_grpnet"``

        Default is ``"grpnet"``.

    family : str, optional
        The family of the response variable.
        It must be one of the following:
        
            - ``"gaussian"``
            - ``"binomial"``
            - ``"poisson"``
            - ``"multigaussian"``
            - ``"multinomial"``

        Default is ``"gaussian"``.
    """

    def __init__(
        self,
        solver: str ="grpnet",
        family: str ="gaussian",
    ):
        """
        Initialize the GroupElasticNet estimator.
        """
        self.solver = solver
        self.family = family

    def fit(
        self, 
        X: Union[np.ndarray, MatrixNaiveBase32, MatrixNaiveBase64], 
        y: np.ndarray, 
        **kwargs: Dict[str, Any],
    ):
        """
        Fit the Group Elastic Net model.

        Parameters
        ----------
        X : (n, p) Union[ndarray, MatrixNaiveBase32, MatrixNaiveBase64]
            Feature matrix.
            It is typically one of the matrices defined in :mod:`adelie.matrix` submodule or :class:`numpy.ndarray`.
        y : (n,) ndarray
            Response vector.
        **kwargs : Dict[str, Any], optional
            Additional arguments to pass to the solver.

        Returns
        -------
        self
            Returns an instance of self.
        """
        self._validate_params()

        # Prepare the response object
        glm_dict = {
            "gaussian": gaussian,
            "binomial": binomial,
            "poisson": poisson,
            "multigaussian": multigaussian,
            "multinomial": multinomial,
        }
        self.glm_ = glm_dict[self.family](y)

        # Choose the solver
        solver_func = {
            "grpnet": grpnet,
            "cv_grpnet": cv_grpnet,
        }[self.solver]

        # Fit the model
        self.state_ = solver_func(
            X=X,
            glm=self.glm_,
            **kwargs,
        )

        # If cross validation used, re-fit with best lambda
        if isinstance(self.state_, CVGrpnetResult):
            self.state_ = self.state_.fit(
                X=X, 
                glm=self.glm_, 
                **kwargs,
            )

            # Store metadata
            self.coef_ = self.state_.betas[-1]
            self.intercept_ = np.array([self.state_.intercepts[-1]])
            self.lambda_ = np.array([self.state_.lmdas[-1]])

        else:
            # Store metadata
            self.coef_ = self.state_.betas
            self.intercept_ = self.state_.intercepts
            self.lambda_ = self.state_.lmdas

        return self

    def predict_proba(
        self, 
        X: Union[np.ndarray, MatrixNaiveBase32, MatrixNaiveBase64], 
    ) -> np.ndarray:
        """
        Predict class probabilities.

        This method is only available for ``"binomial"`` and ``"multinomial"`` families.

        Parameters
        ----------
        X : (n, p) Union[ndarray, MatrixNaiveBase32, MatrixNaiveBase64]
            Feature matrix.
            It is typically one of the matrices defined in :mod:`adelie.matrix` submodule or :class:`numpy.ndarray`.

        Returns
        -------
        proba : ndarray
            The class probabilities at ``X``.
        """
        if not hasattr(self, "state_"):
            raise RuntimeError("The model has not been fitted yet. Call fit() first.")

        if self.family not in ["binomial", "multinomial"]:
            raise ValueError(
                "predict_proba is only available for \"binomial\" and \"multinomial\" families."
            )

        linear_pred = predict(X, self.coef_, self.intercept_)

        if self.family == "binomial":
            proba = expit(linear_pred)
            return np.stack((1 - proba, proba), axis=-1).squeeze()
        elif self.family == "multinomial":
            return softmax(linear_pred, axis=-1).squeeze()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the fitted Group Elastic Net model.

        If ``self.family`` is either ``"binomial"`` or ``"multinomial"``,
        the output is class label predictions based on the largest probability predictions.
        Otherwise, the output is linear predictions.

        Parameters
        ----------
        X : (n, p) Union[ndarray, MatrixNaiveBase32, MatrixNaiveBase64]
            Feature matrix.
            It is typically one of the matrices defined in :mod:`adelie.matrix` submodule or :class:`numpy.ndarray`.

        Returns
        -------
        preds : ndarray
            The class or linear predictions at ``X``.
        """
        if not hasattr(self, "state_"):
            raise RuntimeError("The model has not been fitted yet. Call fit() first.")

        if self.family in ["binomial", "multinomial"]:
            proba = self.predict_proba(X)
            return np.argmax(proba, axis=-1).squeeze()
        else:
            return predict(X, self.coef_, self.intercept_).squeeze()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the R-squared score of the model.

        Parameters
        ----------
        X : (n, p) Union[ndarray, MatrixNaiveBase32, MatrixNaiveBase64]
            Feature matrix.
            It is typically one of the matrices defined in :mod:`adelie.matrix` submodule or :class:`numpy.ndarray`.
        y : (n,) ndarray
            Response vector.

        Returns
        -------
        R2 : float
            The R-squared score.
        """
        yhat = self.predict(X)
        ybar = np.mean(y)
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - ybar) ** 2)
        return np.clip(1 - ss_res / ss_tot, 0, 1)

    def _validate_params(self):
        if self.solver not in ["grpnet", "cv_grpnet"]:
            raise ValueError(f"Unknown solver: {self.solver}")

        if self.family not in [
            "gaussian",
            "binomial",
            "multigaussian",
            "multinomial",
            "poisson",
        ]:
            raise ValueError(f"Unknown family: {self.family}")


class CSSModelSelection(BaseEstimator, RegressorMixin):
    """
    Column Subset Selection estimator for model selection with scikit-learn compatible API.

    The finite-sample guaranteed test procedure for Gaussian features is run
    to identify the smallest subset that most likely reconstructs the rest of the features
    based on the subset factor loss and swapping method.

    Parameters
    ----------
    alpha : float
        Nominal level for the test.
    n_inits : int, optional
        Number of random initializations.
        Default is ``1``.
    n_sims : int, optional
        Number of Monte Carlo samples to estimate critical thresholds.
        Default is ``int(1e4)``.
    n_threads : int, optional
        Number of threads.
        Default is ``1``.
    seed : int, optional
        Random seed.
        If ``None``, no particular seed is used.
        Default is ``None``.

    See Also
    --------
    adelie.solver.css_cov
    """

    def __init__(
        self,
        alpha: float,
        n_inits: int=1,
        n_sims: int=int(1e4),
        n_threads: int =1,
        seed: int =None,
    ):
        """
        Initialize the CSS estimator.
        """
        self.alpha = alpha
        self.n_inits = n_inits
        self.n_sims = n_sims
        self.n_threads = n_threads
        self.seed = seed

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray=None,
    ):
        """
        Fit the CSS model under subset factor loss and perform model selection.

        Parameters
        ----------
        X : (n, p) ndarray
            Feature matrix.
        y : (n,) ndarray, optional
            Not used and only present here for API consistency by convention.
            Default is ``None``.

        Returns
        -------
        self
            Returns an instance of self.

        See Also
        --------
        :func:`fit_cov`
        """
        n = X.shape[0]
        S = X.T @ X / n
        return self.fit_cov(S, n)

    def fit_cov(
        self, 
        S: np.ndarray, 
        n: int,
    ):
        """
        Fit the CSS model under subset factor loss and perform model selection.

        Parameters
        ----------
        S : (p, p) ndarray
            Positive semi-definite matrix :math:`\\Sigma`.
        n : int
            Number of samples.

        Returns
        -------
        self
            Returns an instance of self.

        See Also
        --------
        adelie.solver.css_cov
        """
        alpha = self.alpha
        n_inits = self.n_inits
        n_sims = self.n_sims
        n_threads = self.n_threads
        seed = self.seed
        p = S.shape[1]

        assert p > 0 and n >= p

        # construct covariance estimate
        S = np.asfortranarray(S)
        S_logdet = np.linalg.slogdet(S)[1]

        # prepare samples for critical threshold estimation
        if not (seed is None):
            np.random.seed(seed)
            seeds = np.random.choice(int(1e7), p, replace=False)
        order = np.arange(1, p)
        chi2_1 = np.random.chisquare(order, (n_sims, order.size))
        chi2_2 = np.random.chisquare(n-p-1+order[::-1], (n_sims, order.size))

        # loop through each k and solve CSS
        for k in range(0, p):
            # special case when k == p-1: any subset will not reject
            if k == p-1:
                reject = False
                best_subset = np.arange(p-1)
                break

            # compute (1-alpha) quantile for current k
            numer = chi2_1[:, :(p-k-1)]
            denom = chi2_2[:, (k+1-p):]
            samples = np.sum(np.log(1 + (numer / denom)), axis=-1)
            cutoff = np.quantile(samples, 1-alpha)

            # special case when k == 0: simple computation
            if k == 0:
                T = np.sum(np.log(np.diag(S))) - S_logdet
                reject = T > cutoff
                best_subset = np.empty(0, dtype=int)

            # otherwise, run swapping many times
            else:
                # if k == 1, it is sufficient to run with only 1 initialization.
                n_inits_cap = 1 if k == 1 else n_inits
                assert n_inits_cap >= 1

                # Run n_inits_cap number of random initializations and 
                # solve swapping CSS with subset factor loss.
                # The best (smallest) test statistic is returned along with the best subset.
                fit_out = {
                    np.dtype("float32"): css_cov_model_selection_fit_k_32,
                    np.dtype("float64"): css_cov_model_selection_fit_k_64,
                }[S.dtype](
                    S,
                    k,
                    S_logdet,
                    cutoff,
                    n_inits_cap,
                    n_threads,
                    seeds[k] if seed is None else (p*(k+1) + seed) % 100007,
                )
                best_T = fit_out["T"]
                best_subset = fit_out["subset"]
                reject = best_T > cutoff

            if not reject: break

        self.subset_ = best_subset

        return self

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray=None,
        sample_weight: np.ndarray=None,
    ):
        """Compute the (negative) subset factor loss.

        Parameters
        ----------
        X : (n, p) ndarray
            Feature matrix.
        y : (n,) ndarray, optional
            Not used and only present here for API consistency by convention.
            Default is ``None``.
        sample_weights : (n,) ndarray, optional
            Not used and only present here for API consistency by convention.
            Default is ``None``.

        Returns
        -------
        loss : float
            Subset factor loss where :math:`T` is given by the fitted subset.
        """
        n, p = X.shape
        subset = self.subset_
        subset_c = np.setdiff1d(np.arange(p), subset)
        S_resid = X.T @ X / n
        S_T = np.copy(S_resid[:, subset][subset])
        for i in subset:
            beta = S_resid[i]
            S_resid -= np.outer(beta, beta) / beta[i]
        S_resid_c = S_resid[:, subset_c][subset_c]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loss = np.linalg.slogdet(S_T)[1] + np.sum(np.log(np.diag(S_resid_c)))
        if np.isnan(loss):
            loss = -np.inf
        return -loss