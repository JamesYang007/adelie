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
