"""
antigravity.svm.kernels
=======================
Kernel functions for Support Vector Machines.

All kernels follow the interface::

    K(X, Y, **params) -> ndarray of shape (n_X, n_Y)

where X has shape (n_X, n_features) and Y has shape (n_Y, n_features).

Implemented kernels (Chang & Lin, 2011, Section 2):

* **Linear**      : K(xᵢ, xⱼ) = xᵢᵀ xⱼ
* **Polynomial**  : K(xᵢ, xⱼ) = (γ · xᵢᵀ xⱼ + r)^d
* **RBF**         : K(xᵢ, xⱼ) = exp(−γ · ‖xᵢ − xⱼ‖²)
* **Sigmoid**     : K(xᵢ, xⱼ) = tanh(γ · xᵢᵀ xⱼ + r)

All computations are fully vectorised with NumPy for efficiency.

References
----------
Chang, C.-C., & Lin, C.-J. (2011).
    LIBSVM: A library for support vector machines.
    ACM TIST, 2(3), 27:1–27:27.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Kernel type constants
# ---------------------------------------------------------------------------

KERNEL_LINEAR: str = "linear"
KERNEL_POLY: str = "poly"
KERNEL_RBF: str = "rbf"
KERNEL_SIGMOID: str = "sigmoid"

_VALID_KERNELS = {KERNEL_LINEAR, KERNEL_POLY, KERNEL_RBF, KERNEL_SIGMOID}


# ---------------------------------------------------------------------------
# Individual kernel functions
# ---------------------------------------------------------------------------


def linear_kernel(X: NDArray, Y: NDArray) -> NDArray:
    """Compute the linear kernel matrix between rows of X and Y.

    Formula::

        K(xᵢ, xⱼ) = xᵢᵀ xⱼ

    Parameters
    ----------
    X : ndarray of shape (n_X, n_features)
    Y : ndarray of shape (n_Y, n_features)

    Returns
    -------
    K : ndarray of shape (n_X, n_Y)
        Gram matrix.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    return X @ Y.T


def polynomial_kernel(
    X: NDArray,
    Y: NDArray,
    *,
    gamma: float = 1.0,
    coef0: float = 0.0,
    degree: int = 3,
) -> NDArray:
    """Compute the polynomial kernel matrix between rows of X and Y.

    Formula::

        K(xᵢ, xⱼ) = (γ · xᵢᵀ xⱼ + r)^d

    Parameters
    ----------
    X : ndarray of shape (n_X, n_features)
    Y : ndarray of shape (n_Y, n_features)
    gamma : float, default=1.0
        Scaling factor γ for the inner product.
    coef0 : float, default=0.0
        Independent term r in the kernel.
    degree : int, default=3
        Degree d of the polynomial.

    Returns
    -------
    K : ndarray of shape (n_X, n_Y)
        Gram matrix.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    return (gamma * (X @ Y.T) + coef0) ** degree


def rbf_kernel(X: NDArray, Y: NDArray, *, gamma: float = 1.0) -> NDArray:
    """Compute the Radial Basis Function (RBF / Gaussian) kernel matrix.

    Formula::

        K(xᵢ, xⱼ) = exp(−γ · ‖xᵢ − xⱼ‖²)

    The squared Euclidean distance is expanded as::

        ‖xᵢ − xⱼ‖² = ‖xᵢ‖² + ‖xⱼ‖² − 2 xᵢᵀ xⱼ

    to exploit fast matrix multiplication (O(n² d) instead of O(n² d²)).

    Parameters
    ----------
    X : ndarray of shape (n_X, n_features)
    Y : ndarray of shape (n_Y, n_features)
    gamma : float, default=1.0
        Bandwidth parameter γ. Larger values → tighter bell.

    Returns
    -------
    K : ndarray of shape (n_X, n_Y)
        Gram matrix with all entries in (0, 1].
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    # Squared row norms
    X_sq = np.einsum("ij,ij->i", X, X)          # shape (n_X,)
    Y_sq = np.einsum("ij,ij->i", Y, Y)          # shape (n_Y,)
    cross = X @ Y.T                              # shape (n_X, n_Y)

    # ‖xᵢ − xⱼ‖² via broadcast; clip avoids tiny negatives from FP errors
    sq_dists = np.clip(X_sq[:, None] + Y_sq[None, :] - 2.0 * cross, 0.0, None)
    return np.exp(-gamma * sq_dists)


def sigmoid_kernel(
    X: NDArray,
    Y: NDArray,
    *,
    gamma: float = 1.0,
    coef0: float = 0.0,
) -> NDArray:
    """Compute the sigmoid kernel matrix between rows of X and Y.

    Formula::

        K(xᵢ, xⱼ) = tanh(γ · xᵢᵀ xⱼ + r)

    .. warning::
        The sigmoid kernel is **not** positive semi-definite for all
        parameter choices, so the QP solver may encounter indefinite
        sub-problems.  Use a small jitter (``tau``) in the solver for
        numerical safety.

    Parameters
    ----------
    X : ndarray of shape (n_X, n_features)
    Y : ndarray of shape (n_Y, n_features)
    gamma : float, default=1.0
        Scaling factor γ.
    coef0 : float, default=0.0
        Independent term r.

    Returns
    -------
    K : ndarray of shape (n_X, n_Y)
        Gram matrix.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    return np.tanh(gamma * (X @ Y.T) + coef0)


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------


def compute_kernel_matrix(
    X: NDArray,
    Y: NDArray,
    kernel: str = KERNEL_RBF,
    *,
    gamma: float | None = None,
    coef0: float = 0.0,
    degree: int = 3,
) -> NDArray:
    """Dispatch to the requested kernel and compute the full Gram matrix.

    Parameters
    ----------
    X : ndarray of shape (n_X, n_features)
    Y : ndarray of shape (n_Y, n_features)
    kernel : {'linear', 'poly', 'rbf', 'sigmoid'}, default='rbf'
        Which kernel to use.
    gamma : float or None, default=None
        Kernel coefficient for 'poly', 'rbf', and 'sigmoid'.
        If ``None``, defaults to ``1 / n_features``.
    coef0 : float, default=0.0
        Independent term for 'poly' and 'sigmoid'.
    degree : int, default=3
        Degree for the polynomial kernel.

    Returns
    -------
    K : ndarray of shape (n_X, n_Y)

    Raises
    ------
    ValueError
        If *kernel* is not one of the supported values.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if kernel not in _VALID_KERNELS:
        raise ValueError(
            f"Unknown kernel '{kernel}'. Choose from {sorted(_VALID_KERNELS)}."
        )

    # Default γ = 1 / n_features (standard LIBSVM convention)
    if gamma is None:
        gamma = 1.0 / X.shape[1] if X.shape[1] > 0 else 1.0

    if kernel == KERNEL_LINEAR:
        return linear_kernel(X, Y)
    if kernel == KERNEL_POLY:
        return polynomial_kernel(X, Y, gamma=gamma, coef0=coef0, degree=degree)
    if kernel == KERNEL_RBF:
        return rbf_kernel(X, Y, gamma=gamma)
    # KERNEL_SIGMOID
    return sigmoid_kernel(X, Y, gamma=gamma, coef0=coef0)


# ---------------------------------------------------------------------------
# Kernel diagonal utility (used by the solver for Q_ii)
# ---------------------------------------------------------------------------


def kernel_diagonal(
    X: NDArray,
    kernel: str = KERNEL_RBF,
    *,
    gamma: float | None = None,
    coef0: float = 0.0,
    degree: int = 3,
) -> NDArray:
    """Return the diagonal of K(X, X) without computing the full matrix.

    For **RBF** and **linear** kernels this can be done in O(n·d) instead
    of O(n²·d).  For polynomial and sigmoid we still compute the diagonal
    via vectorised einsum.

    Parameters
    ----------
    X : ndarray of shape (n, n_features)
    kernel, gamma, coef0, degree : same as :func:`compute_kernel_matrix`.

    Returns
    -------
    diag : ndarray of shape (n,)
    """
    X = np.asarray(X, dtype=np.float64)

    if gamma is None:
        gamma = 1.0 / X.shape[1] if X.shape[1] > 0 else 1.0

    if kernel == KERNEL_LINEAR:
        return np.einsum("ij,ij->i", X, X)

    if kernel == KERNEL_RBF:
        # K(xᵢ, xᵢ) = exp(0) = 1  for all i
        return np.ones(X.shape[0], dtype=np.float64)

    if kernel == KERNEL_POLY:
        dot_ii = np.einsum("ij,ij->i", X, X)
        return (gamma * dot_ii + coef0) ** degree

    if kernel == KERNEL_SIGMOID:
        dot_ii = np.einsum("ij,ij->i", X, X)
        return np.tanh(gamma * dot_ii + coef0)

    raise ValueError(f"Unknown kernel '{kernel}'.")
