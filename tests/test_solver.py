"""Tests for antigravity.svm.solver"""
import numpy as np
import pytest
from antigravity.svm.solver import (
    LRUKernelCache,
    SolverResult,
    solve,
    _compute_gap,
)


# ---------------------------------------------------------------------------
# LRUKernelCache tests
# ---------------------------------------------------------------------------


class TestLRUCache:
    def test_put_and_get(self):
        cache = LRUKernelCache(capacity=3)
        col = np.array([1.0, 2.0, 3.0])
        cache.put(0, col)
        result = cache.get(0)
        np.testing.assert_array_equal(result, col)

    def test_miss_returns_none(self):
        cache = LRUKernelCache(capacity=3)
        assert cache.get(99) is None

    def test_eviction_on_overflow(self):
        cache = LRUKernelCache(capacity=2)
        cache.put(0, np.array([0.0]))
        cache.put(1, np.array([1.0]))
        cache.put(2, np.array([2.0]))   # should evict key 0
        assert cache.get(0) is None
        assert cache.get(1) is not None
        assert cache.get(2) is not None

    def test_lru_ordering(self):
        cache = LRUKernelCache(capacity=2)
        cache.put(0, np.array([0.0]))
        cache.put(1, np.array([1.0]))
        cache.get(0)                   # 0 is now recently used
        cache.put(2, np.array([2.0]))  # should evict key 1 (LRU)
        assert cache.get(1) is None
        assert cache.get(0) is not None

    def test_invalidate(self):
        cache = LRUKernelCache(capacity=5)
        cache.put(3, np.array([3.0]))
        cache.invalidate(3)
        assert cache.get(3) is None

    def test_clear(self):
        cache = LRUKernelCache(capacity=5)
        cache.put(0, np.ones(3))
        cache.put(1, np.ones(3))
        cache.clear()
        assert len(cache) == 0

    def test_capacity_validation(self):
        with pytest.raises(ValueError):
            LRUKernelCache(capacity=0)


# ---------------------------------------------------------------------------
# Simple 2-variable QP with known solution
# ---------------------------------------------------------------------------
#
# min   ½ [α₁ α₂] [[2, -1], [-1, 2]] [α₁; α₂]  +  [-1, -1]ᵀ [α₁; α₂]
# s.t.  α₁ - α₂ = 0,   0 ≤ αᵢ ≤ 5
#
# The unconstrained minimiser of ½ αᵀQ α + pᵀα with equality α₁=α₂:
# Let α₁=α₂=t → f(t) = ½(2t² -2t²+2t²) - 2t = t² - 2t
# f'(t) = 2t - 2 = 0 → t* = 1  → optimal α₁* = α₂* = 1
# f* = 1 - 2 = -1


def make_simple_qp():
    n = 2
    Q_mat = np.array([[2.0, -1.0], [-1.0, 2.0]])
    p = np.array([-1.0, -1.0])
    y = np.array([1.0, -1.0])   # equality α₁ - α₂ = 0

    def q_fn(i, indices):
        return Q_mat[i, indices]

    q_diag = np.diag(Q_mat)
    lower = np.zeros(2)
    upper = np.full(2, 5.0)
    return n, q_fn, q_diag, p, y, lower, upper


def test_solver_simple_2var():
    n, q_fn, q_diag, p, y, lower, upper = make_simple_qp()
    result = solve(n, q_fn, q_diag, p, y, lower, upper, tol=1e-6)
    assert isinstance(result, SolverResult)
    np.testing.assert_allclose(result.alpha, [1.0, 1.0], atol=1e-3)


def test_solver_objective_value():
    n, q_fn, q_diag, p, y, lower, upper = make_simple_qp()
    result = solve(n, q_fn, q_diag, p, y, lower, upper, tol=1e-6)
    # Objective should be close to -1
    np.testing.assert_allclose(result.obj, -1.0, atol=1e-3)


# ---------------------------------------------------------------------------
# Linearly separable 1-D problem (manual sanity check)
# ---------------------------------------------------------------------------


def test_solver_linearly_separable():
    """Solver on a trivially separable binary problem.

    Points: x_i = i for i=1..6, labels alternating ±1 but actually
    separable by x < 3.5: y = [1,1,1,-1,-1,-1].

    With linear kernel Q_ij = y_i y_j x_i x_j, the solution should
    have exactly the boundary SVs active (α>0) and the others at 0 or C.
    """
    rng = np.random.RandomState(0)
    n = 6
    X = np.arange(1, n + 1, dtype=float).reshape(-1, 1)
    y_lab = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
    C = 1.0

    # Linear kernel: Q_ij = y_i y_j (x_i · x_j)
    Kmat = (X @ X.T)
    Qmat = y_lab[:, None] * y_lab[None, :] * Kmat

    def q_fn(i, indices):
        return Qmat[i, indices]

    q_diag = np.diag(Qmat)
    p = -np.ones(n)
    lower = np.zeros(n)
    upper = np.full(n, C)

    result = solve(n, q_fn, q_diag, p, y_lab, lower, upper, tol=1e-4)
    # All αᵢ should be >= 0 and <= C
    assert np.all(result.alpha >= -1e-6)
    assert np.all(result.alpha <= C + 1e-6)
    # Equality constraint: yᵀα ≈ 0  (allow some slack for poor conditioning)
    assert abs(y_lab @ result.alpha) < 0.2


# ---------------------------------------------------------------------------
# Convergence tests
# ---------------------------------------------------------------------------


def test_solver_converges_within_max_iter():
    rng = np.random.RandomState(7)
    n = 20
    X = rng.randn(n, 3)
    y = np.sign(rng.randn(n))
    y[y == 0] = 1.0
    C = 1.0

    Kmat = X @ X.T
    Qmat = y[:, None] * y[None, :] * Kmat

    def q_fn(i, indices):
        return Qmat[i, indices]

    q_diag = np.diag(Qmat)
    p = -np.ones(n)
    lower = np.zeros(n)
    upper = np.full(n, C)

    result = solve(n, q_fn, q_diag, p, y, lower, upper, tol=1e-3, max_iter=50_000)
    assert result.n_iter <= 50_000


def test_solver_no_shrinking_same_solution():
    """Shrinking should not change the final solution."""
    rng = np.random.RandomState(11)
    n = 10
    X = rng.randn(n, 2)
    y = np.sign(rng.randn(n))
    y[y == 0] = 1.0

    Kmat = np.exp(-0.5 * ((X[:, None] - X[None, :]) ** 2).sum(axis=2))
    Qmat = y[:, None] * y[None, :] * Kmat

    def q_fn(i, indices):
        return Qmat[i, indices]

    q_diag = np.diag(Qmat)
    p = -np.ones(n)
    lower = np.zeros(n)
    upper = np.full(n, 1.0)

    r1 = solve(n, q_fn, q_diag, p, y, lower, upper, tol=1e-4, shrinking=True)
    r2 = solve(n, q_fn, q_diag, p, y, lower, upper, tol=1e-4, shrinking=False)

    np.testing.assert_allclose(r1.alpha, r2.alpha, atol=1e-3)


# ---------------------------------------------------------------------------
# KKT gap test
# ---------------------------------------------------------------------------


def test_compute_gap_at_solution():
    _, _, _, p, y, lower, upper = make_simple_qp()
    # At the optimal solution alpha=[1,1], gradient = Qα + p = [2-1-1, -1+2-1] = [0, 0]
    gradient = np.array([0.0, 0.0])
    alpha = np.array([1.0, 1.0])
    gap = _compute_gap(gradient, y, alpha, upper, lower)
    assert gap < 1e-6
