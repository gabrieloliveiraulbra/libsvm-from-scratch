"""Tests for antigravity.svm.kernels"""
import numpy as np
import pytest
from antigravity.svm.kernels import (
    linear_kernel,
    polynomial_kernel,
    rbf_kernel,
    sigmoid_kernel,
    compute_kernel_matrix,
    kernel_diagonal,
    KERNEL_LINEAR,
    KERNEL_POLY,
    KERNEL_RBF,
    KERNEL_SIGMOID,
)


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def XY(rng):
    X = rng.randn(10, 4)
    Y = rng.randn(8, 4)
    return X, Y


# --- Shape correctness ---

def test_linear_shape(XY):
    X, Y = XY
    K = linear_kernel(X, Y)
    assert K.shape == (10, 8)


def test_poly_shape(XY):
    X, Y = XY
    K = polynomial_kernel(X, Y, gamma=0.5, coef0=1.0, degree=3)
    assert K.shape == (10, 8)


def test_rbf_shape(XY):
    X, Y = XY
    K = rbf_kernel(X, Y, gamma=0.5)
    assert K.shape == (10, 8)


def test_sigmoid_shape(XY):
    X, Y = XY
    K = sigmoid_kernel(X, Y, gamma=0.5, coef0=0.1)
    assert K.shape == (10, 8)


# --- Symmetry ---

def test_linear_symmetric(rng):
    X = rng.randn(6, 4)
    K = linear_kernel(X, X)
    np.testing.assert_allclose(K, K.T, atol=1e-12)


def test_rbf_symmetric(rng):
    X = rng.randn(6, 4)
    K = rbf_kernel(X, X, gamma=0.3)
    np.testing.assert_allclose(K, K.T, atol=1e-12)


def test_poly_symmetric(rng):
    X = rng.randn(6, 4)
    K = polynomial_kernel(X, X, gamma=0.5, coef0=1.0, degree=2)
    np.testing.assert_allclose(K, K.T, atol=1e-12)


# --- RBF diagonal = 1 ---

def test_rbf_diagonal_ones(rng):
    X = rng.randn(10, 5)
    K = rbf_kernel(X, X, gamma=1.0)
    np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-12)


# --- RBF values in (0, 1] ---

def test_rbf_range(XY):
    X, Y = XY
    K = rbf_kernel(X, Y, gamma=0.5)
    assert np.all(K > 0)
    assert np.all(K <= 1 + 1e-12)


# --- PSD for RBF ---

def test_rbf_psd(rng):
    X = rng.randn(12, 3)
    K = rbf_kernel(X, X, gamma=0.5)
    eigvals = np.linalg.eigvalsh(K)
    assert np.all(eigvals >= -1e-10), f"Negative eigenvalue: {eigvals.min()}"


# --- Linear kernel matches manual dot-product ---

def test_linear_matches_dot(XY):
    X, Y = XY
    K = linear_kernel(X, Y)
    expected = X @ Y.T
    np.testing.assert_allclose(K, expected, atol=1e-12)


# --- kernel_diagonal matches compute_kernel_matrix diagonal ---

@pytest.mark.parametrize("kernel", [KERNEL_LINEAR, KERNEL_POLY, KERNEL_RBF, KERNEL_SIGMOID])
def test_kernel_diagonal(rng, kernel):
    X = rng.randn(8, 3)
    gamma = 0.5
    diag_fast = kernel_diagonal(X, kernel, gamma=gamma, coef0=0.5, degree=2)
    K_full = compute_kernel_matrix(X, X, kernel, gamma=gamma, coef0=0.5, degree=2)
    np.testing.assert_allclose(diag_fast, np.diag(K_full), atol=1e-10)


# --- compute_kernel_matrix dispatches correctly ---

def test_compute_kernel_matrix_dispatch(XY):
    X, Y = XY
    for kernel in [KERNEL_LINEAR, KERNEL_POLY, KERNEL_RBF, KERNEL_SIGMOID]:
        K = compute_kernel_matrix(X, Y, kernel, gamma=0.5)
        assert K.shape == (10, 8)


def test_unknown_kernel(XY):
    X, Y = XY
    with pytest.raises(ValueError, match="Unknown kernel"):
        compute_kernel_matrix(X, Y, "cosine")


# --- Default gamma = 1/n_features ---

def test_default_gamma(rng):
    X = rng.randn(5, 4)
    K1 = compute_kernel_matrix(X, X, KERNEL_RBF, gamma=None)
    K2 = compute_kernel_matrix(X, X, KERNEL_RBF, gamma=1.0 / 4)
    np.testing.assert_allclose(K1, K2, atol=1e-12)
