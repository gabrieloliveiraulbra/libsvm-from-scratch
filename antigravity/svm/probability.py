"""
antigravity.svm.probability
===========================
Probability calibration for SVM outputs.

Two estimators are provided (Chang & Lin, 2011, Section 8):

:class:`PlattScaling`
    Maps binary SVM decision values to class probabilities via a fitted
    sigmoid (Platt, 1999)::

        P(y = +1 | f) = 1 / (1 + exp(A·f + B))

    Parameters ``(A, B)`` are estimated using 5-fold cross-validation
    on the training data to avoid overfitting, then fitted with
    Newton's method (Lin, Lin & Weng, 2007).

:class:`LaplaceSVR`
    Estimates the Laplace scale σ for SVR prediction intervals via
    cross-validation on the absolute residuals::

        |y − f(x)|  ~  Laplace(0, σ)  →  σ̂ = (1/n) Σ |yᵢ − fᵢ|

    Provides ``predict_interval`` and ``predict_proba``.

References
----------
Platt, J. (1999).
    Probabilistic outputs for support vector machines and comparisons
    to regularized likelihood methods. Advances in Large Margin
    Classifiers, 10(3), 61–74.
Lin, H.-T., Lin, C.-J., & Weng, R. C. (2007).
    A note on Platt's probabilistic outputs for support vector machines.
    Machine Learning, 68(3), 267–276.
Chang, C.-C., & Lin, C.-J. (2011). LIBSVM. ACM TIST, 2(3), 27.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Platt Scaling (Classification)
# ---------------------------------------------------------------------------


class PlattScaling:
    """Sigmoid calibration of SVM decision values.

    Fits ``P(y=+1 | f) = σ(A·f + B)`` where σ is the logistic function.
    The optimal ``(A, B)`` minimise the cross-entropy over the
    cross-validated decision values via Newton's method.

    Parameters
    ----------
    n_folds : int, default=5
        Number of cross-validation folds used to generate unbiased
        decision values for fitting the sigmoid.
    max_iter : int, default=100
        Maximum Newton iterations for optimising ``(A, B)``.
    tol : float, default=1e-5
        Newton convergence tolerance on the gradient norm.

    Attributes
    ----------
    A_ : float
        Sigmoid slope (negative for well-calibrated SVMs).
    B_ : float
        Sigmoid intercept.
    """

    def __init__(
        self,
        n_folds: int = 5,
        max_iter: int = 100,
        tol: float = 1e-5,
    ) -> None:
        self.n_folds = n_folds
        self.max_iter = max_iter
        self.tol = tol
        self.A_: float = 0.0
        self.B_: float = 0.0

    # ------------------------------------------------------------------
    def fit(self, decision_values: NDArray, y: NDArray) -> "PlattScaling":
        """Fit ``(A, B)`` from decision values and true binary labels.

        Uses the method of Lin, Lin & Weng (2007) with prior-probability
        label smoothing to prevent overfitting.

        Parameters
        ----------
        decision_values : ndarray of shape (n,)
            Raw SVM decision values ``f(xᵢ)`` for each training sample.
        y : ndarray of shape (n,)
            True labels encoded as +1 / -1.

        Returns
        -------
        self
        """
        f = np.asarray(decision_values, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_pos = int((y > 0).sum())
        n_neg = int((y < 0).sum())

        # Label smoothing (prior regularisation)
        t = np.where(
            y > 0,
            (n_pos + 1.0) / (n_pos + 2.0),
            1.0 / (n_neg + 2.0),
        )

        self.A_, self.B_ = _fit_sigmoid_newton(f, t, self.max_iter, self.tol)
        return self

    # ------------------------------------------------------------------
    def fit_with_cv(
        self,
        model_factory,
        X: NDArray,
        y: NDArray,
    ) -> "PlattScaling":
        """Fit via cross-validation on the training data.

        The *model_factory* callable must return a freshly constructed
        (unfitted) SVM model with a ``.decision_function`` method.

        Parameters
        ----------
        model_factory : callable () -> unfitted SVM model
            Called once per fold to create a fresh model instance.
        X : ndarray of shape (n, d)
        y : ndarray of shape (n,)  — ±1 labels.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n = len(y)
        n_folds = self.n_folds

        decision_oos = np.zeros(n)
        indices = np.arange(n)
        np.random.shuffle(indices)
        folds = np.array_split(indices, n_folds)

        for fold in folds:
            val_idx = fold
            train_idx = np.concatenate([f for f in folds if f is not fold])
            m = model_factory()
            m.fit(X[train_idx], y[train_idx])
            decision_oos[val_idx] = m.decision_function(X[val_idx])

        self.fit(decision_oos, y)
        return self

    # ------------------------------------------------------------------
    def predict_proba(self, decision_values: NDArray) -> NDArray:
        """Convert decision values to class probabilities.

        Parameters
        ----------
        decision_values : ndarray of shape (n,)

        Returns
        -------
        proba : ndarray of shape (n, 2)
            ``[:, 0]`` = P(y=−1 | f),  ``[:, 1]`` = P(y=+1 | f).
        """
        f = np.asarray(decision_values, dtype=np.float64)
        fApB = f * self.A_ + self.B_
        # Numerically stable sigmoid
        p_pos = np.where(
            fApB >= 0,
            np.exp(-fApB) / (1.0 + np.exp(-fApB)),
            1.0 / (1.0 + np.exp(fApB)),
        )
        return np.column_stack([1.0 - p_pos, p_pos])

    # ------------------------------------------------------------------
    def predict_proba_binary(self, decision_values: NDArray) -> NDArray:
        """Return P(y=+1 | f) as a 1-D array."""
        return self.predict_proba(decision_values)[:, 1]


# ---------------------------------------------------------------------------
# Newton's method for sigmoid fitting (Lin, Lin & Weng, 2007)
# ---------------------------------------------------------------------------


def _fit_sigmoid_newton(
    f: NDArray,
    t: NDArray,
    max_iter: int = 100,
    tol: float = 1e-5,
) -> Tuple[float, float]:
    """Fit sigmoid parameters (A, B) via Newton's method.

    Minimises the binary cross-entropy::

        L(A, B) = − Σᵢ [ tᵢ log(pᵢ) + (1 − tᵢ) log(1 − pᵢ) ]

    where ``pᵢ = σ(A·fᵢ + B)`` and ``σ`` is the logistic function.

    Parameters
    ----------
    f : ndarray of shape (n,) — decision values.
    t : ndarray of shape (n,) — smoothed target probabilities in (0, 1).
    max_iter : int
    tol : float

    Returns
    -------
    (A, B) : Tuple[float, float]
    """
    n = len(f)
    A = 0.0
    B = math.log((np.sum(t <= 0) + 1.0) / (np.sum(t > 0) + 1.0))
    # B initialised to log(ratio of negative to positive targets)

    _MINLOG = 1e-300   # clamp for log stability

    for _ in range(max_iter):
        fApB = f * A + B

        # pᵢ = P(y=+1 | fᵢ),  qᵢ = 1 − pᵢ
        # Numerically stable for both large-positive and large-negative fApB
        p = np.where(fApB >= 0, np.exp(-fApB) / (1 + np.exp(-fApB)),
                     1.0 / (1 + np.exp(fApB)))
        q = 1.0 - p

        # Hessian diagonal components
        h11 = np.dot(f * f, p * q) + 1e-10
        h22 = np.sum(p * q) + 1e-10
        h21 = np.dot(f, p * q)

        # Gradient
        g1 = np.dot(f, t - p)
        g2 = np.sum(t - p)

        # Stop if gradient is small
        if abs(g1) < tol and abs(g2) < tol:
            break

        # Newton step: solve H·d = −g  (2×2 system)
        det = h11 * h22 - h21 * h21
        if abs(det) < 1e-20:
            break
        dA = -(h22 * g1 - h21 * g2) / det
        dB = -(h11 * g2 - h21 * g1) / det

        # Armijo line-search
        step = 1.0
        loss_old = _log_loss(f, t, A, B, _MINLOG)
        while step > 1e-10:
            A_new = A + step * dA
            B_new = B + step * dB
            loss_new = _log_loss(f, t, A_new, B_new, _MINLOG)
            if loss_new < loss_old:
                A, B = A_new, B_new
                break
            step *= 0.5
        else:
            break   # line-search failed to improve

    return A, B


def _log_loss(
    f: NDArray, t: NDArray, A: float, B: float, minlog: float
) -> float:
    """Compute cross-entropy loss for (A, B)."""
    fApB = f * A + B
    p = np.where(fApB >= 0, np.exp(-fApB) / (1 + np.exp(-fApB)),
                 1.0 / (1 + np.exp(fApB)))
    p = np.clip(p, minlog, 1 - minlog)
    return float(-np.dot(t, np.log(p)) - np.dot(1 - t, np.log(1 - p)))


# ---------------------------------------------------------------------------
# Laplace SVR Probability Estimation
# ---------------------------------------------------------------------------


class LaplaceSVR:
    """Probability estimation for SVR using a Laplace residual model.

    For a trained SVR with out-of-sample predictions ``f(xᵢ)``, the
    absolute residuals ``|yᵢ − f(xᵢ)|`` are modelled as an exponential
    (half-Laplace) distribution with scale ``σ``::

        P(|y − f(x)| ≤ ε) = 1 − exp(−ε / σ)

    The scale is estimated as the mean absolute error on out-of-sample
    predictions (via cross-validation)::

        σ̂ = (1 / n) Σᵢ |yᵢ − fᵢ|

    Parameters
    ----------
    n_folds : int, default=5
        Number of CV folds for computing out-of-sample residuals.

    Attributes
    ----------
    sigma_ : float
        Estimated Laplace scale parameter.
    """

    def __init__(self, n_folds: int = 5) -> None:
        self.n_folds = n_folds
        self.sigma_: float = 1.0

    # ------------------------------------------------------------------
    def fit(self, residuals: NDArray) -> "LaplaceSVR":
        """Fit the scale from precomputed out-of-sample residuals.

        Parameters
        ----------
        residuals : ndarray of shape (n,)
            Absolute residuals ``|yᵢ − f(xᵢ)|``.

        Returns
        -------
        self
        """
        r = np.asarray(residuals, dtype=np.float64)
        self.sigma_ = float(np.mean(np.abs(r)))
        if self.sigma_ < 1e-12:
            self.sigma_ = 1e-12   # safety clamp
        return self

    # ------------------------------------------------------------------
    def fit_with_cv(
        self,
        model_factory,
        X: NDArray,
        y: NDArray,
    ) -> "LaplaceSVR":
        """Estimate σ via cross-validation.

        Parameters
        ----------
        model_factory : callable () -> unfitted SVR model
        X : ndarray of shape (n, d)
        y : ndarray of shape (n,) — regression targets.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n = len(y)
        n_folds = self.n_folds

        pred_oos = np.zeros(n)
        indices = np.arange(n)
        np.random.shuffle(indices)
        folds = np.array_split(indices, n_folds)

        for fold in folds:
            val_idx = fold
            train_idx = np.concatenate([f for f in folds if f is not fold])
            m = model_factory()
            m.fit(X[train_idx], y[train_idx])
            pred_oos[val_idx] = m.predict(X[val_idx])

        self.fit(y - pred_oos)
        return self

    # ------------------------------------------------------------------
    def predict_interval(
        self, y_pred: NDArray, confidence: float = 0.95
    ) -> Tuple[NDArray, NDArray]:
        """Compute a prediction interval for each point.

        Given ``P(|y − f(x)| ≤ ε) = 1 − exp(−ε / σ) = confidence``,
        solving for ε::

            ε = −σ · log(1 − confidence)

        Parameters
        ----------
        y_pred : ndarray of shape (n,) — SVR predictions.
        confidence : float in (0, 1), default=0.95

        Returns
        -------
        (lower, upper) : Tuple[ndarray, ndarray]
            Each array has shape (n,).
        """
        if not 0 < confidence < 1:
            raise ValueError("confidence must be in (0, 1).")
        eps = -self.sigma_ * math.log(1.0 - confidence)
        lower = y_pred - eps
        upper = y_pred + eps
        return lower, upper

    # ------------------------------------------------------------------
    def predict_proba(self, y_pred: NDArray, y_true: NDArray) -> NDArray:
        """Estimate P(|y − f(x)| ≤ |y_true − y_pred|) for each sample.

        Parameters
        ----------
        y_pred : ndarray of shape (n,)
        y_true : ndarray of shape (n,)

        Returns
        -------
        proba : ndarray of shape (n,) — values in [0, 1).
        """
        residuals = np.abs(
            np.asarray(y_true, dtype=np.float64) -
            np.asarray(y_pred, dtype=np.float64)
        )
        return 1.0 - np.exp(-residuals / self.sigma_)
