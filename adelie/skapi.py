from typing import Dict, Any
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from .glm import binomial, gaussian, poisson, multigaussian, multinomial
from .solver import grpnet
from .cv import cv_grpnet, CVGrpnetResult
from .diagnostic import predict


class GroupElasticNet(BaseEstimator, RegressorMixin):
    """
    Group Elastic Net estimator with scikit-learn compatible API.
    """

    def __init__(
        self,
        solver: str = "grpnet",
        family: str = "gaussian",
    ):
        """
        Initialize the GroupElasticNet estimator.

        Args:
            solver (str): The solver to use. Either "grpnet" or "cv_grpnet".
            family (str): The family of the response variable.
        """
        self.solver = solver
        self.family = family

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Dict[str, Any]):
        """
        Fit the Group Elastic Net model.

        Args:
            X (np.ndarray): The input samples.
            y (np.ndarray): The target values.
            **kwargs: Additional arguments to pass to the solver.

        Returns:
            self: Returns an instance of self.
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
        solver_func = cv_grpnet if self.solver == "cv_grpnet" else grpnet

        # Fit the model
        self.state_ = solver_func(
            X=X,
            glm=self.glm_,
            **kwargs,
        )

        # If cross validation used, re-fit with best lambda
        if isinstance(self.state_, CVGrpnetResult):
            lm_star = self.state_.lmdas[self.state_.best_idx]
            self.lmda_star_ = lm_star
            # Final state is a model with single lambda (and hence single coef)
            self.state_ = grpnet(
                X=X,
                glm=self.glm_,
                lmda_path=np.r_[lm_star:lm_star:1j],  # only one lambda
                **kwargs,
            )

        # Store coefficients and intercepts
        self.coef_ = self.state_.betas
        self.intercept_ = self.state_.intercepts

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the fitted Group Elastic Net model.

        Args:
            X (np.ndarray): The input samples.

        Returns:
            np.ndarray: The predicted values.
        """
        if not hasattr(self, "state_"):
            raise RuntimeError("The model has not been fitted yet. Call 'fit' first.")
        return predict(X, self.coef_, self.intercept_).squeeze()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the R-squared score of the model.

        Args:
            X (np.ndarray): The input samples.
            y (np.ndarray): The true values.

        Returns:
            float: The R-squared score.
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
            "poisson",
            "multigaussian",
            "multinomial",
        ]:
            raise ValueError(f"Unknown family: {self.family}")
