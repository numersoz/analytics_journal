import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class RecursiveLeastSquares(BaseEstimator, RegressorMixin):
    """
    Recursive Least Squares (RLS) estimator for online linear regression.

    This algorithm estimates parameters of the model:

        y_t = x_tᵀ θ + ε_t

    in a recursive fashion as new data arrives. It minimizes the exponentially 
    weighted least-squares cost:

        J_t(θ) = Σ_{τ=1}^t λ^{t-τ} (y_τ - x_τᵀ θ)^2

    The update equations for each new observation (x_t, y_t) are:

        ε_t     = y_t - x_tᵀ θ_{t-1}
        K_t     = P_{t-1} x_t / (λ + x_tᵀ P_{t-1} x_t)
        θ_t     = θ_{t-1} + K_t ε_t
        P_t     = (P_{t-1} - K_t x_tᵀ P_{t-1}) / λ

    Attributes:
        theta_ (np.ndarray): Current parameter estimate (n_features, 1).
        P_ (np.ndarray): Inverse covariance matrix (n_features, n_features).
    """

    def __init__(self, n_features: int, delta: float = 1e3, lambda_: float = 1.0):
        """
        Initializes the RLS model.

        Args:
            n_features (int): Number of features in the input data.
            delta (float): Initial value for P = δ·I (large for high uncertainty).
            lambda_ (float): Forgetting factor in (0, 1]; λ=1 gives no forgetting.
        """
        self.n_features = n_features
        self.delta = delta
        self.lambda_ = lambda_
        self._initialized = False

    def _initialize(self):
        """Initializes the parameter vector and covariance matrix."""
        self.theta_ = np.zeros((self.n_features, 1))
        self.P_ = self.delta * np.eye(self.n_features)
        self._initialized = True

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> "RecursiveLeastSquares":
        """
        Performs recursive updates on the model using new data.

        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features).
            y (np.ndarray): Target values of shape (n_samples,).

        Returns:
            RecursiveLeastSquares: Self, updated with the new data.
        """
        if not self._initialized:
            self._initialize()

        X = np.atleast_2d(X)
        y = np.atleast_1d(y)

        for x_t, y_t in zip(X, y):
            x_t = x_t.reshape(-1, 1)
            y_t = np.array(y_t).reshape(1, 1)

            # ε_t = y_t - x_tᵀ θ_{t-1}
            epsilon_t = y_t - x_t.T @ self.theta_

            # K_t = P_{t-1} x_t / (λ + x_tᵀ P_{t-1} x_t)
            denominator = float(self.lambda_ + x_t.T @ self.P_ @ x_t)
            K_t = self.P_ @ x_t / denominator

            # θ_t = θ_{t-1} + K_t ε_t
            self.theta_ = self.theta_ + K_t @ epsilon_t

            # P_t = (P_{t-1} - K_t x_tᵀ P_{t-1}) / λ
            self.P_ = (self.P_ - K_t @ x_t.T @ self.P_) / self.lambda_

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts target values for given input features.

        Args:
            X (np.ndarray): Input array of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted values of shape (n_samples,).
        """
        if not self._initialized:
            raise ValueError("Model is not initialized. Call `partial_fit` first.")

        return (X @ self.theta_).flatten()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RecursiveLeastSquares":
        """
        Fits the model to data using repeated recursive updates.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): Target values.

        Returns:
            RecursiveLeastSquares: Self.
        """
        return self.partial_fit(X, y)

    def reset(self):
        """Re-initializes the model to forget past data."""
        self._initialize()
