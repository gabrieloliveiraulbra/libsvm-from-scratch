"""
antigravity.svm.models
======================
SVM model classes for the five formulations described in Chang & Lin (2011).

All models share a consistent scikit-learn-compatible interface::

    model.fit(X, y)          -> self
    model.predict(X)         -> ndarray
    model.decision_function(X) -> ndarray

Models
------
:class:`CSVC`
    C-Support Vector Classification (binary, Section 1.1).
:class:`NuSVC`
    ν-Support Vector Classification (binary, Section 1.2).
:class:`OneClassSVM`
    One-Class SVM for novelty/outlier detection (Section 1.3).
:class:`EpsilonSVR`
    ε-Support Vector Regression (Section 1.4).
:class:`NuSVR`
    ν-Support Vector Regression (Section 1.5).

Internal helper :func:`_build_q_fn` constructs the column-fetching closure
that feeds into :func:`~antigravity.svm.solver.solve`.

References
----------
Chang, C.-C., & Lin, C.-J. (2011).
    LIBSVM: A library for support vector machines. ACM TIST, 2(3), 27.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from antigravity.svm.kernels import (
    compute_kernel_matrix,
    kernel_diagonal,
    KERNEL_RBF,
)
from antigravity.svm.solver import solve, SolverResult

# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


def _default_gamma(X: NDArray, gamma: Optional[float]) -> float:
    """Return gamma, defaulting to 1/n_features if None."""
    if gamma is not None:
        return gamma
    return 1.0 / X.shape[1] if X.shape[1] > 0 else 1.0


def _build_q_fn(
    X: NDArray,
    y: NDArray,
    kernel: str,
    gamma: float,
    coef0: float,
    degree: int,
    sign: NDArray,
) -> "tuple[callable, NDArray]":
    """Build the Q-column function and its diagonal.

    Q_ij = sign_i · sign_j · K(x_i, x_j)

    where *sign* encodes label information:
    - For classification: sign = y  (±1)
    - For regression ε-SVR: sign encodes [+1,...,+1,−1,...,−1]
    - For ν-SVR / One-Class: sign = all +1 (Gram matrix is Q itself)

    Returns
    -------
    q_fn : callable(i, indices) -> ndarray
    q_diag : ndarray of shape (n,)
    """

    def q_fn(i: int, indices: NDArray) -> NDArray:
        """Return column i of Q for the given row indices."""
        xi = X[i : i + 1]           # shape (1, d)
        xj = X[indices]             # shape (len(indices), d)
        k_vals = compute_kernel_matrix(
            xi, xj, kernel, gamma=gamma, coef0=coef0, degree=degree
        ).ravel()   # shape (len(indices),)
        return sign[i] * sign[indices] * k_vals

    q_diag = kernel_diagonal(X, kernel, gamma=gamma, coef0=coef0, degree=degree)
    q_diag = sign ** 2 * q_diag   # sign²=1 for ±1 labels; generalises correctly

    return q_fn, q_diag


def _compute_decision_function(
    X_train: NDArray,
    X_test: NDArray,
    alpha: NDArray,
    y_train: NDArray,
    rho: float,
    kernel: str,
    gamma: float,
    coef0: float,
    degree: int,
) -> NDArray:
    """Evaluate the SVM decision function on X_test.

    f(x) = Σᵢ αᵢ yᵢ K(xᵢ, x) − ρ

    Parameters
    ----------
    X_train : ndarray of shape (n_sv, d) — support vectors (all training pts).
    X_test  : ndarray of shape (m, d).
    alpha   : ndarray of shape (n_sv,) — dual coefficients.
    y_train : ndarray of shape (n_sv,) — labels used to build Q (±1).
    rho     : float — bias term.

    Returns
    -------
    decision : ndarray of shape (m,)
    """
    K = compute_kernel_matrix(
        X_test, X_train, kernel, gamma=gamma, coef0=coef0, degree=degree
    )   # shape (m, n_sv)
    return K @ (alpha * y_train) - rho


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class _SVMBase:
    """Internal base class with shared kernel / solver wiring."""

    def __init__(
        self,
        kernel: str = KERNEL_RBF,
        gamma: Optional[float] = None,
        coef0: float = 0.0,
        degree: int = 3,
        tol: float = 1e-3,
        max_iter: int = -1,
        cache_size: int = 500,
        shrinking: bool = True,
        verbose: bool = False,
    ) -> None:
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.tol = tol
        self.max_iter = max_iter
        self.cache_size = cache_size
        self.shrinking = shrinking
        self.verbose = verbose

        # Fitted attributes (set by .fit())
        self.support_vectors_: Optional[NDArray] = None
        self.support_: Optional[NDArray] = None          # indices into training X
        self.dual_coef_: Optional[NDArray] = None        # αᵢ · yᵢ for SVCs
        self.intercept_: Optional[float] = None          # −ρ
        self._alpha: Optional[NDArray] = None
        self._rho: Optional[float] = None
        self._X_train: Optional[NDArray] = None
        self._y_train: Optional[NDArray] = None
        self._gamma_fit: Optional[float] = None
        self._result: Optional[SolverResult] = None

    # ------------------------------------------------------------------
    def _resolve_max_iter(self, n: int) -> int:
        if self.max_iter > 0:
            return self.max_iter
        return max(10_000, 10 * n)

    # ------------------------------------------------------------------
    def _store_sv(self, X: NDArray, y: NDArray, alpha: NDArray) -> None:
        """Extract and store support vectors and dual coefficients."""
        sv_mask = alpha > 0
        self.support_ = np.where(sv_mask)[0]
        self.support_vectors_ = X[sv_mask]
        # dual_coef = αᵢ · yᵢ  (convention matching sklearn)
        self.dual_coef_ = (alpha[sv_mask] * y[sv_mask]).reshape(1, -1)


# ---------------------------------------------------------------------------
# C-SVC
# ---------------------------------------------------------------------------


class CSVC(_SVMBase):
    """C-Support Vector Classification.

    Solves the dual QP (Chang & Lin, 2011, Eq. 1)::

        min   ½ αᵀ Q α − eᵀ α
        s.t.  yᵀ α = 0,   0 ≤ αᵢ ≤ C

    where Q_ij = yᵢ yⱼ K(xᵢ, xⱼ) and e is the all-ones vector.

    Decision function::

        f(x) = sign( Σᵢ αᵢ yᵢ K(xᵢ, x) − ρ )

    Parameters
    ----------
    C : float, default=1.0
        Regularisation parameter.  Smaller C → wider margin / more misclassifications.
    kernel : {'linear', 'poly', 'rbf', 'sigmoid'}, default='rbf'
    gamma : float or None
        Kernel coefficient. Defaults to 1/n_features.
    coef0 : float, default=0.0
        Free term for polynomial and sigmoid kernels.
    degree : int, default=3
        Degree of the polynomial kernel.
    tol : float, default=1e-3
        KKT violation tolerance.
    max_iter : int, default=-1
        Maximum solver iterations.  -1 means auto (max(10000, 10·n)).
    cache_size : int, default=500
        Number of kernel columns to cache.
    shrinking : bool, default=True
        Use shrinking heuristic.
    verbose : bool, default=False
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = KERNEL_RBF,
        gamma: Optional[float] = None,
        coef0: float = 0.0,
        degree: int = 3,
        tol: float = 1e-3,
        max_iter: int = -1,
        cache_size: int = 500,
        shrinking: bool = True,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            kernel=kernel, gamma=gamma, coef0=coef0, degree=degree,
            tol=tol, max_iter=max_iter, cache_size=cache_size,
            shrinking=shrinking, verbose=verbose,
        )
        self.C = C
        self.classes_: Optional[NDArray] = None

    # ------------------------------------------------------------------
    def fit(self, X: NDArray, y: NDArray) -> "CSVC":
        """Train the C-SVC classifier.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,)
            Class labels.  Only two distinct values are supported (binary).

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(
                f"CSVC expects exactly 2 classes, got {len(self.classes_)}. "
                "Use MulticlassSVC for multi-class problems."
            )

        # Map labels to ±1
        y_bin = np.where(y == self.classes_[1], 1.0, -1.0)

        n = len(y_bin)
        gamma = _default_gamma(X, self.gamma)
        self._gamma_fit = gamma

        # Build Q-column function: Q_ij = y_i y_j K(x_i, x_j)
        q_fn, q_diag = _build_q_fn(
            X, y_bin, self.kernel, gamma, self.coef0, self.degree, sign=y_bin
        )

        # p = −e  (maximise Σαᵢ  ⟺  minimise −Σαᵢ)
        p = -np.ones(n)
        lower = np.zeros(n)
        upper = np.full(n, self.C)

        result = solve(
            n=n, Q_fn=q_fn, Q_diag=q_diag,
            p=p, y=y_bin, lower=lower, upper=upper,
            tol=self.tol, max_iter=self._resolve_max_iter(n),
            cache_size=self.cache_size, shrinking=self.shrinking,
            verbose=self.verbose,
        )

        self._result = result
        self._alpha = result.alpha
        self._rho = result.rho
        self._X_train = X
        self._y_train = y_bin
        self.intercept_ = -result.rho
        self._store_sv(X, y_bin, result.alpha)
        return self

    # ------------------------------------------------------------------
    def decision_function(self, X: NDArray) -> NDArray:
        """Compute the raw decision values f(x) for each sample.

        Parameters
        ----------
        X : ndarray of shape (m, n_features)

        Returns
        -------
        decision : ndarray of shape (m,)
            Positive → class 1, negative → class 0.
        """
        if self._X_train is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        X = np.asarray(X, dtype=np.float64)
        return _compute_decision_function(
            self._X_train, X,
            self._alpha, self._y_train, self._rho,
            self.kernel, self._gamma_fit, self.coef0, self.degree,
        )

    # ------------------------------------------------------------------
    def predict(self, X: NDArray) -> NDArray:
        """Predict class labels for X.

        Parameters
        ----------
        X : ndarray of shape (m, n_features)

        Returns
        -------
        y_pred : ndarray of shape (m,)
            Original class labels (not ±1).
        """
        decisions = self.decision_function(X)
        binary = np.where(decisions >= 0, 1, -1)
        return np.where(binary == 1, self.classes_[1], self.classes_[0])


# ---------------------------------------------------------------------------
# ν-SVC
# ---------------------------------------------------------------------------


class NuSVC(_SVMBase):
    """ν-Support Vector Classification.

    Solves the scaled dual (Chang & Lin, 2011, Section 1.2, Eq. 3)::

        min   ½ ᾱᵀ Q ᾱ
        s.t.  yᵀ ᾱ = 0,   eᵀ ᾱ = ν
              0 ≤ ᾱᵢ ≤ 1/l

    The parameter ν ∈ (0, 1] is an upper bound on the fraction of
    margin errors and a lower bound on the fraction of support vectors.

    Parameters
    ----------
    nu : float, default=0.5
        ν parameter in (0, 1].
    kernel, gamma, coef0, degree, tol, max_iter, cache_size,
    shrinking, verbose : same as :class:`CSVC`.
    """

    def __init__(
        self,
        nu: float = 0.5,
        kernel: str = KERNEL_RBF,
        gamma: Optional[float] = None,
        coef0: float = 0.0,
        degree: int = 3,
        tol: float = 1e-3,
        max_iter: int = -1,
        cache_size: int = 500,
        shrinking: bool = True,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            kernel=kernel, gamma=gamma, coef0=coef0, degree=degree,
            tol=tol, max_iter=max_iter, cache_size=cache_size,
            shrinking=shrinking, verbose=verbose,
        )
        self.nu = nu
        self.classes_: Optional[NDArray] = None

    # ------------------------------------------------------------------
    def fit(self, X: NDArray, y: NDArray) -> "NuSVC":
        """Train the ν-SVC.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)  — binary labels.

        Returns
        -------
        self

        Notes
        -----
        The ν-SVC dual is equivalent to C-SVC with per-class upper bounds
        chosen to enforce the fraction constraint (Crisp & Burges, 2000;
        LIBSVM svm.cpp, solve_nu_svc):

            C_pos = ν · n / (2 · n_pos),  C_neg = ν · n / (2 · n_neg)
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("NuSVC expects exactly 2 classes.")

        y_bin = np.where(y == self.classes_[1], 1.0, -1.0)
        n = len(y_bin)
        gamma = _default_gamma(X, self.gamma)
        self._gamma_fit = gamma

        n_pos = float((y_bin > 0).sum())
        n_neg = float((y_bin < 0).sum())
        if n_pos == 0 or n_neg == 0:
            raise ValueError("NuSVC requires at least one sample of each class.")

        # ν-SVC ⇔ C-SVC with per-class upper bounds (LIBSVM solve_nu_svc)
        C_pos = self.nu * n / (2.0 * n_pos)
        C_neg = self.nu * n / (2.0 * n_neg)
        upper = np.where(y_bin > 0, C_pos, C_neg)

        q_fn, q_diag = _build_q_fn(
            X, y_bin, self.kernel, gamma, self.coef0, self.degree, sign=y_bin
        )
        # p = -ones (same as C-SVC) creates proper descent direction
        p = -np.ones(n)
        lower = np.zeros(n)

        result = solve(
            n=n, Q_fn=q_fn, Q_diag=q_diag,
            p=p, y=y_bin, lower=lower, upper=upper,
            tol=self.tol, max_iter=self._resolve_max_iter(n),
            cache_size=self.cache_size, shrinking=self.shrinking,
            verbose=self.verbose,
        )

        self._result = result
        self._alpha = result.alpha
        self._rho = result.rho
        self._X_train = X
        self._y_train = y_bin
        self.intercept_ = -result.rho
        self._store_sv(X, y_bin, result.alpha)
        return self

    # ------------------------------------------------------------------
    def decision_function(self, X: NDArray) -> NDArray:
        """Raw decision values."""
        if self._X_train is None:
            raise RuntimeError("Call .fit() first.")
        X = np.asarray(X, dtype=np.float64)
        return _compute_decision_function(
            self._X_train, X,
            self._alpha, self._y_train, self._rho,
            self.kernel, self._gamma_fit, self.coef0, self.degree,
        )

    # ------------------------------------------------------------------
    def predict(self, X: NDArray) -> NDArray:
        """Predict class labels."""
        decisions = self.decision_function(X)
        binary = np.where(decisions >= 0, 1, -1)
        return np.where(binary == 1, self.classes_[1], self.classes_[0])


# ---------------------------------------------------------------------------
# One-Class SVM
# ---------------------------------------------------------------------------


class OneClassSVM(_SVMBase):
    """One-Class SVM for novelty / outlier detection.

    Solves (Schölkopf et al., 2001; Chang & Lin, 2011, Section 1.3)::

        min   ½ αᵀ K α
        s.t.  eᵀ α = 1,   0 ≤ αᵢ ≤ 1/(ν·n)

    where K is the plain kernel matrix (no label sign).

    A new point x is classified as *inlier* (+1) if::

        f(x) = Σᵢ αᵢ K(xᵢ, x) − ρ ≥ 0

    Parameters
    ----------
    nu : float, default=0.5
        Upper bound on the fraction of outliers and lower bound on
        the fraction of support vectors.
    kernel, gamma, coef0, degree, tol, max_iter, cache_size,
    shrinking, verbose : same as :class:`CSVC`.
    """

    def __init__(
        self,
        nu: float = 0.5,
        kernel: str = KERNEL_RBF,
        gamma: Optional[float] = None,
        coef0: float = 0.0,
        degree: int = 3,
        tol: float = 1e-3,
        max_iter: int = -1,
        cache_size: int = 500,
        shrinking: bool = True,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            kernel=kernel, gamma=gamma, coef0=coef0, degree=degree,
            tol=tol, max_iter=max_iter, cache_size=cache_size,
            shrinking=shrinking, verbose=verbose,
        )
        self.nu = nu

    # ------------------------------------------------------------------
    def fit(self, X: NDArray, y: NDArray = None) -> "OneClassSVM":
        """Train the One-Class SVM.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ignored (present for API consistency)

        Returns
        -------
        self

        Notes
        -----
        The One-Class SVM dual (Schölkopf et al., 2001)::

            min  ½ αᵀ K α
            s.t. eᵀ α = 1,   0 ≤ αᵢ ≤ 1/(ν·n)

        is converted to a C-SVC-style QP using n+1 variables:
        the first n are the real αᵢ (label y=+1, cost p=-1) and the
        (n+1)-th is a sentinel variable (label y=−1, kernel=0, cost p=0)
        whose dual absorbs the equality so that yᵀz = 0 is satisfied.
        This maps the constraint eᵀα = 1 to the standard yᵀz = 0 form.
        """
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        gamma = _default_gamma(X, self.gamma)
        self._gamma_fit = gamma

        # Augmented (n+1)-variable problem:
        #   y_aug = [+1, ..., +1, -1]   (sentinel has label -1)
        #   p_aug = [-1, ..., -1,  0]   (drive real α upward; sentinel neutral)
        #   upper = [C_real, ..., C_real, 1]  where C_real = 1/(ν·n)
        #   Q sentinel row/col = 0  (virtual origin with zero kernel values)
        # The equality yᵀz = 0 ⟺ Σα_real = α_sentinel, which combined with
        # driving α_real up and leaving α_sentinel free realises the one-class
        # sum constraint at optimum.
        n_aug = n + 1
        y_aug = np.concatenate([np.ones(n), [-1.0]])
        p_aug = np.concatenate([-np.ones(n), [0.0]])
        lower_aug = np.zeros(n_aug)
        C_real = 1.0 / (self.nu * n)
        upper_aug = np.concatenate([np.full(n, C_real), [1.0]])

        # Q diagonal: K(xᵢ, xᵢ) for reals, 0 for sentinel
        k_diag_real = kernel_diagonal(
            X, self.kernel, gamma=gamma, coef0=self.coef0, degree=self.degree
        )
        q_diag_aug = np.concatenate([k_diag_real, [0.0]])

        def q_fn_aug(i: int, indices: NDArray) -> NDArray:
            """Column i of Q for the (n+1)-variable augmented problem."""
            col = np.zeros(len(indices))
            real_mask = indices < n
            if i < n:
                # Real sample: Q_{i,j} = y_i·y_j·K(xᵢ,xⱼ) = K(xᵢ,xⱼ)  for j < n
                #              Q_{i, sentinel} = y_i·(-1)·0 = 0
                xi = X[i : i + 1]
                xj_real = X[indices[real_mask]]
                if len(xj_real) > 0:
                    k_vals = compute_kernel_matrix(
                        xi, xj_real, self.kernel,
                        gamma=gamma, coef0=self.coef0, degree=self.degree
                    ).ravel()
                    col[real_mask] = k_vals   # y_i=+1, y_j=+1 → Q=K
            # else: sentinel row is all zeros
            return col

        result = solve(
            n=n_aug, Q_fn=q_fn_aug, Q_diag=q_diag_aug,
            p=p_aug, y=y_aug, lower=lower_aug, upper=upper_aug,
            tol=self.tol, max_iter=self._resolve_max_iter(n),
            cache_size=self.cache_size, shrinking=self.shrinking,
            verbose=self.verbose,
        )

        # The real one-class αᵢ are the first n variables
        alpha_eff = result.alpha[:n]

        # Compute ρ directly from the kernel sum at free support vectors.
        # For a free SV (0 < αᵢ < C_real): f(xᵢ) = Σⱼ αⱼ K(xⱼ, xᵢ) − ρ = 0
        # → ρ = mean(Σⱼ αⱼ K(xⱼ, xᵢ))  over free SVs.
        eps_sv = 1e-8
        free_mask = (alpha_eff > eps_sv) & (alpha_eff < C_real - eps_sv)
        if free_mask.any():
            K_free = compute_kernel_matrix(
                X[free_mask], X, self.kernel,
                gamma=gamma, coef0=self.coef0, degree=self.degree,
            )   # shape (n_free, n)
            rho = float(np.mean(K_free @ alpha_eff))
        else:
            # Fall back: use mean over all SVs
            sv_mask = alpha_eff > eps_sv
            if sv_mask.any():
                K_sv = compute_kernel_matrix(
                    X[sv_mask], X, self.kernel,
                    gamma=gamma, coef0=self.coef0, degree=self.degree,
                )
                rho = float(np.mean(K_sv @ alpha_eff))
            else:
                rho = 0.0

        self._result = result
        self._alpha = alpha_eff
        self._rho = rho
        self._X_train = X
        self._y_train = np.ones(n)
        self.intercept_ = -rho
        self._store_sv(X, np.ones(n), alpha_eff)
        return self


    # ------------------------------------------------------------------
    def decision_function(self, X: NDArray) -> NDArray:
        """Compute f(x) = Σαᵢ K(xᵢ, x) − ρ.

        Parameters
        ----------
        X : ndarray of shape (m, n_features)

        Returns
        -------
        scores : ndarray of shape (m,)
        """
        if self._X_train is None:
            raise RuntimeError("Call .fit() first.")
        X = np.asarray(X, dtype=np.float64)
        return _compute_decision_function(
            self._X_train, X,
            self._alpha, self._y_train, self._rho,
            self.kernel, self._gamma_fit, self.coef0, self.degree,
        )

    # ------------------------------------------------------------------
    def predict(self, X: NDArray) -> NDArray:
        """Predict +1 (inlier) or -1 (outlier) for each sample.

        Parameters
        ----------
        X : ndarray of shape (m, n_features)

        Returns
        -------
        y_pred : ndarray of shape (m,) with values in {-1, +1}.
        """
        return np.where(self.decision_function(X) >= 0, 1, -1).astype(np.int8)


# ---------------------------------------------------------------------------
# ε-SVR
# ---------------------------------------------------------------------------


class EpsilonSVR(_SVMBase):
    """ε-Support Vector Regression.

    Solves the dual (Chang & Lin, 2011, Eq. 5) by substituting
    β = α − α* and working on 2l variables::

        min   ½ β̃ᵀ Q̃ β̃ + ε eᵀ(α + α*) − yᵀ(α − α*)
        s.t.  eᵀ(α − α*) = 0,   0 ≤ αᵢ, αᵢ* ≤ C

    Internally re-ordered as a single vector of length 2l::

        z = [α₁, …, α_l, α₁*, …, α_l*]
        ỹ = [+1, …, +1, −1, …, −1]

    so that ỹᵀ z = 0 is the equality constraint, and

        Q̃_{ij} = ỹᵢ ỹⱼ K(x_{i mod l}, x_{j mod l})

    Parameters
    ----------
    C : float, default=1.0
        Regularisation parameter.
    epsilon : float, default=0.1
        Width of the insensitive tube.
    kernel, gamma, coef0, degree, tol, max_iter, cache_size,
    shrinking, verbose : same as :class:`CSVC`.
    """

    def __init__(
        self,
        C: float = 1.0,
        epsilon: float = 0.1,
        kernel: str = KERNEL_RBF,
        gamma: Optional[float] = None,
        coef0: float = 0.0,
        degree: int = 3,
        tol: float = 1e-3,
        max_iter: int = -1,
        cache_size: int = 500,
        shrinking: bool = True,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            kernel=kernel, gamma=gamma, coef0=coef0, degree=degree,
            tol=tol, max_iter=max_iter, cache_size=cache_size,
            shrinking=shrinking, verbose=verbose,
        )
        self.C = C
        self.epsilon = epsilon

    # ------------------------------------------------------------------
    def fit(self, X: NDArray, y: NDArray) -> "EpsilonSVR":
        """Train the ε-SVR.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,)  — regression targets.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n = len(y)
        gamma = _default_gamma(X, self.gamma)
        self._gamma_fit = gamma

        # 2l-variable dual encoding: z = [α; α*], ỹ = [+1…+1, −1…−1]
        n2 = 2 * n
        y2 = np.concatenate([np.ones(n), -np.ones(n)])

        # p_i = ε − y_i  for α_i;  p_i = ε + y_i  for α_i*
        p = np.concatenate([self.epsilon - y, self.epsilon + y])
        lower = np.zeros(n2)
        upper = np.full(n2, self.C)

        # Q-column function for 2l-variable problem:
        # Q_ij = ỹ_i ỹ_j K(x_{i%l}, x_{j%l})
        def q_fn(i: int, indices: NDArray) -> NDArray:
            xi = X[i % n : (i % n) + 1]
            xj = X[indices % n]
            k_vals = compute_kernel_matrix(
                xi, xj, self.kernel,
                gamma=gamma, coef0=self.coef0, degree=self.degree
            ).ravel()
            return y2[i] * y2[indices] * k_vals

        # Diagonal: Q_ii = K(x_{i%l}, x_{i%l})
        k_diag = kernel_diagonal(
            X, self.kernel, gamma=gamma, coef0=self.coef0, degree=self.degree
        )
        q_diag = np.concatenate([k_diag, k_diag])   # y²=1

        result = solve(
            n=n2, Q_fn=q_fn, Q_diag=q_diag,
            p=p, y=y2, lower=lower, upper=upper,
            tol=self.tol, max_iter=self._resolve_max_iter(n2),
            cache_size=self.cache_size, shrinking=self.shrinking,
            verbose=self.verbose,
        )

        # Recover α − α* for prediction
        alpha_full = result.alpha
        alpha_pos = alpha_full[:n]
        alpha_neg = alpha_full[n:]
        coef = alpha_pos - alpha_neg   # shape (n,)

        self._result = result
        self._alpha = coef            # net dual coefficients
        self._rho = result.rho
        self._X_train = X
        self._y_train = np.ones(n)   # sign is already in coef
        self.intercept_ = -result.rho
        self.dual_coef_ = coef.reshape(1, -1)

        # Support vectors: those with |αᵢ − αᵢ*| > 0
        sv_mask = np.abs(coef) > 1e-10
        self.support_ = np.where(sv_mask)[0]
        self.support_vectors_ = X[sv_mask]
        return self

    # ------------------------------------------------------------------
    def predict(self, X: NDArray) -> NDArray:
        """Predict regression targets for X.

        f(x) = Σᵢ (αᵢ − αᵢ*) K(xᵢ, x) − ρ

        Parameters
        ----------
        X : ndarray of shape (m, n_features)

        Returns
        -------
        y_pred : ndarray of shape (m,)
        """
        if self._X_train is None:
            raise RuntimeError("Call .fit() first.")
        X = np.asarray(X, dtype=np.float64)
        K = compute_kernel_matrix(
            X, self._X_train, self.kernel,
            gamma=self._gamma_fit, coef0=self.coef0, degree=self.degree,
        )
        return K @ self._alpha - self._rho

    # ------------------------------------------------------------------
    def decision_function(self, X: NDArray) -> NDArray:
        """Alias for :meth:`predict` (regression has no separate decision fn)."""
        return self.predict(X)


# ---------------------------------------------------------------------------
# ν-SVR
# ---------------------------------------------------------------------------


class NuSVR(_SVMBase):
    """ν-Support Vector Regression.

    Solves the dual with modified upper bounds (Chang & Lin, 2011, Eq. 6)::

        min   ½ βᵀ Q β + yᵀ β
        s.t.  eᵀ β = 0,   0 ≤ βᵢ ≤ C/l

    encoded as a 2l variable problem analogous to ε-SVR, but with ε
    determined automatically by ν.

    Parameters
    ----------
    C : float, default=1.0
        Regularisation parameter.
    nu : float, default=0.5
        ν parameter controlling the fraction of training errors and SVs.
    kernel, gamma, coef0, degree, tol, max_iter, cache_size,
    shrinking, verbose : same as :class:`CSVC`.
    """

    def __init__(
        self,
        C: float = 1.0,
        nu: float = 0.5,
        kernel: str = KERNEL_RBF,
        gamma: Optional[float] = None,
        coef0: float = 0.0,
        degree: int = 3,
        tol: float = 1e-3,
        max_iter: int = -1,
        cache_size: int = 500,
        shrinking: bool = True,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            kernel=kernel, gamma=gamma, coef0=coef0, degree=degree,
            tol=tol, max_iter=max_iter, cache_size=cache_size,
            shrinking=shrinking, verbose=verbose,
        )
        self.C = C
        self.nu = nu

    # ------------------------------------------------------------------
    def fit(self, X: NDArray, y: NDArray) -> "NuSVR":
        """Train the ν-SVR.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,)  — regression targets.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n = len(y)
        gamma = _default_gamma(X, self.gamma)
        self._gamma_fit = gamma

        # ν-SVR: same 2l encoding as ε-SVR but with C_i = C/n for all vars,
        # and p = [y; -y] (maximise ε tube radius implicitly)
        # LIBSVM (svm.cpp, solve_nu_svr) sets upper = C * nu / 2 for α
        # and uses epsilon determined internally.
        n2 = 2 * n
        y2 = np.concatenate([np.ones(n), -np.ones(n)])

        # ν-SVR dual: p = [y; -y] (negate to minimise objective)
        p = np.concatenate([-y, y])
        C_eff = self.C * self.nu / 2.0
        lower = np.zeros(n2)
        # Upper bound: each α and α* is bounded by C_eff = C·ν/2
        upper = np.full(n2, C_eff)

        def q_fn(i: int, indices: NDArray) -> NDArray:
            xi = X[i % n : (i % n) + 1]
            xj = X[indices % n]
            k_vals = compute_kernel_matrix(
                xi, xj, self.kernel,
                gamma=gamma, coef0=self.coef0, degree=self.degree
            ).ravel()
            return y2[i] * y2[indices] * k_vals

        k_diag = kernel_diagonal(
            X, self.kernel, gamma=gamma, coef0=self.coef0, degree=self.degree
        )
        q_diag = np.concatenate([k_diag, k_diag])

        result = solve(
            n=n2, Q_fn=q_fn, Q_diag=q_diag,
            p=p, y=y2, lower=lower, upper=upper,
            tol=self.tol, max_iter=self._resolve_max_iter(n2),
            cache_size=self.cache_size, shrinking=self.shrinking,
            verbose=self.verbose,
        )

        alpha_pos = result.alpha[:n]
        alpha_neg = result.alpha[n:]
        coef = alpha_pos - alpha_neg

        self._result = result
        self._alpha = coef
        self._rho = result.rho
        self._X_train = X
        self._y_train = np.ones(n)
        self.intercept_ = -result.rho
        self.dual_coef_ = coef.reshape(1, -1)

        sv_mask = np.abs(coef) > 1e-10
        self.support_ = np.where(sv_mask)[0]
        self.support_vectors_ = X[sv_mask]
        return self

    # ------------------------------------------------------------------
    def predict(self, X: NDArray) -> NDArray:
        """Predict regression targets for X."""
        if self._X_train is None:
            raise RuntimeError("Call .fit() first.")
        X = np.asarray(X, dtype=np.float64)
        K = compute_kernel_matrix(
            X, self._X_train, self.kernel,
            gamma=self._gamma_fit, coef0=self.coef0, degree=self.degree,
        )
        return K @ self._alpha - self._rho

    # ------------------------------------------------------------------
    def decision_function(self, X: NDArray) -> NDArray:
        """Alias for :meth:`predict`."""
        return self.predict(X)
