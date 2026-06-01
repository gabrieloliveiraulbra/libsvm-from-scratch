"""
antigravity.svm.solver
======================
Sequential Minimal Optimisation (SMO) solver for the general SVM QP problem.

Solves the Quadratic Programme::

    min   ½ αᵀ Q α + pᵀ α
    s.t.  yᵀ α = 0
          lᵢ ≤ αᵢ ≤ uᵢ,  i = 1, …, l

where Q is the (possibly indefinite) kernel-weighted Hessian and p, y, l, u
are problem-specific vectors.

Key sub-components
------------------
:class:`LRUKernelCache`
    Fixed-size LRU cache for columns of Q, avoiding redundant re-evaluation.

:func:`select_working_set`
    Second-order WSS-3 working set selection (Fan, Chen & Lin, 2005) choosing
    the pair (i, j) that gives the greatest predicted decrease in the objective.

:func:`solve`
    Main SMO loop with shrinking and KKT-convergence checking.

References
----------
* Chang, C.-C., & Lin, C.-J. (2011). LIBSVM: A library for support
  vector machines. ACM TIST, 2(3), 27:1–27:27.
* Fan, R.-E., Chen, P.-H., & Lin, C.-J. (2005). Working set selection
  using second-order information for training SVMs. JMLR, 6, 1889–1918.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Callable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

KernelFn = Callable[[NDArray, NDArray], NDArray]  # (X_i, X_j) -> float matrix

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TAU: float = 1e-12   # minimum denominator for step-size clipping (PSD jitter)
_INF: float = math.inf


# ---------------------------------------------------------------------------
# LRU Kernel Cache
# ---------------------------------------------------------------------------


class LRUKernelCache:
    """Cache for columns of the kernel matrix Q.

    Each column ``j`` is ``Q[:, j]`` restricted to the **active** set.
    The cache uses an :class:`collections.OrderedDict` for O(1) LRU
    bookkeeping.

    Parameters
    ----------
    capacity : int
        Maximum number of columns to hold in memory.

    Notes
    -----
    The cache stores **full-length** columns (length = number of training
    samples).  Callers are responsible for re-computing columns when the
    active set changes during shrinking.
    """

    def __init__(self, capacity: int = 500) -> None:
        if capacity < 1:
            raise ValueError("Cache capacity must be >= 1.")
        self._capacity = capacity
        self._store: OrderedDict[int, NDArray] = OrderedDict()

    # ------------------------------------------------------------------
    def get(self, key: int) -> Optional[NDArray]:
        """Return cached column *key* and mark it as recently used.

        Returns ``None`` if not present.
        """
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    # ------------------------------------------------------------------
    def put(self, key: int, column: NDArray) -> None:
        """Insert *column* under *key*, evicting the LRU entry if full."""
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = column
        if len(self._store) > self._capacity:
            self._store.popitem(last=False)   # remove least-recently-used

    # ------------------------------------------------------------------
    def invalidate(self, key: int) -> None:
        """Remove a single key if present (used after shrinking changes)."""
        self._store.pop(key, None)

    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Flush the entire cache."""
        self._store.clear()

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# SolverResult dataclass-like container
# ---------------------------------------------------------------------------


class SolverResult:
    """Container for the output of :func:`solve`.

    Attributes
    ----------
    alpha : ndarray of shape (n_samples,)
        Optimal dual variables.
    obj : float
        Optimal primal objective value ½ αᵀ Q α + pᵀ α.
    rho : float
        The optimal threshold (bias term) ρ in the decision function.
    n_iter : int
        Number of SMO iterations performed.
    r_sq : float
        Feasibility residual ‖yᵀ α‖ at termination.
    """

    __slots__ = ("alpha", "obj", "rho", "n_iter", "r_sq")

    def __init__(
        self,
        alpha: NDArray,
        obj: float,
        rho: float,
        n_iter: int,
        r_sq: float,
    ) -> None:
        self.alpha = alpha
        self.obj = obj
        self.rho = rho
        self.n_iter = n_iter
        self.r_sq = r_sq

    def __repr__(self) -> str:
        return (
            f"SolverResult(obj={self.obj:.6g}, rho={self.rho:.6g}, "
            f"n_iter={self.n_iter}, n_sv={int((self.alpha > 0).sum())})"
        )


# ---------------------------------------------------------------------------
# Working Set Selection — WSS 3 (Fan et al., 2005)
# ---------------------------------------------------------------------------


def _select_working_set(
    gradient: NDArray,
    y: NDArray,
    alpha: NDArray,
    C: NDArray,
    Q_diag: NDArray,
    get_column: Callable[[int], NDArray],
    active: NDArray,
) -> Tuple[int, int]:
    """Select the working pair (i, j) using second-order information (WSS-3).

    Algorithm (Fan, Chen & Lin, 2005, Algorithm 3):

    1. **Select i**: the index in I_up that maximises ``-y·∇f``:

       .. code-block::

          I_up = {t : α_t < C_t  and  y_t = +1}
               ∪ {t : α_t > 0   and  y_t = -1}
          i = argmax_{t ∈ I_up} (-y_t · ∇f_t)

    2. **Select j**: minimise the second-order decrease::

          min_{t ∈ I_low, -y_t·∇f_t < -y_i·∇f_i}
              [ - (∇f_i - ∇f_t)² / (Q_ii + Q_tt - 2 Q_it) ]

    Parameters
    ----------
    gradient : ndarray of shape (l,)
        Current gradient ∇f = Qα + p (full length, active indices only used).
    y : ndarray of shape (l,)  — ±1 labels.
    alpha : ndarray of shape (l,)  — current dual variables.
    C : ndarray of shape (l,)  — per-sample upper bounds.
    Q_diag : ndarray of shape (l,)  — diagonal of Q.
    get_column : callable (int) -> ndarray of shape (l,)
        Retrieve column i of Q (from cache or recompute).
    active : ndarray of bool, shape (l,)
        Mask of non-shrunken variables.

    Returns
    -------
    (i, j) : Tuple[int, int]
        Global indices of the selected working set pair.
        Returns (-1, -1) if no violating pair exists (converged).
    """
    active_idx = np.where(active)[0]
    g = gradient[active_idx]
    y_a = y[active_idx]
    a = alpha[active_idx]
    C_a = C[active_idx]
    Qd = Q_diag[active_idx]

    # --- Compute -y·∇f for all active variables ---
    m_yg = -y_a * g  # shape (len_active,)

    # I_up mask: can increase α (not yet at upper bound in the y-direction)
    I_up = ((y_a > 0) & (a < C_a)) | ((y_a < 0) & (a > 0))
    # I_low mask: can decrease α
    I_low = ((y_a > 0) & (a > 0)) | ((y_a < 0) & (a < C_a))

    if not I_up.any() or not I_low.any():
        return -1, -1

    # --- Step 1: select i ---
    m_yg_up = np.where(I_up, m_yg, -_INF)
    local_i = int(np.argmax(m_yg_up))
    max_mygf = m_yg_up[local_i]
    i = int(active_idx[local_i])

    # --- Step 2: select j with second-order information ---
    # Retrieve full column i of Q restricted to active set
    col_i = get_column(i)[active_idx]   # shape (len_active,)

    # Denominator: Q_ii + Q_tt - 2 Q_it  (≥ τ for stability)
    denom = Qd[local_i] + Qd - 2.0 * col_i
    denom = np.maximum(denom, _TAU)

    # Numerator squared: (∇f_i - ∇f_t)²  ==  (-y·∇f_i - -y·∇f_t)² / y²
    # Since y = ±1, y² = 1, so just (max_mygf - m_yg)²:
    num_sq = (max_mygf - m_yg) ** 2

    # Objective decrease (negative means real decrease); only consider I_low
    # with -y_t·∇f_t < max_mygf (otherwise direction is infeasible / no gain)
    feasible = I_low & (m_yg < max_mygf)
    if not feasible.any():
        return -1, -1

    gain = np.where(feasible, -num_sq / denom, _INF)
    local_j = int(np.argmin(gain))
    j = int(active_idx[local_j])

    return i, j


# ---------------------------------------------------------------------------
# Gradient reconstruction for shrinking
# ---------------------------------------------------------------------------


def _reconstruct_gradient(
    gradient: NDArray,
    alpha: NDArray,
    alpha_status: NDArray,
    p: NDArray,
    y: NDArray,
    C: NDArray,
    get_column: Callable[[int], NDArray],
    n: int,
) -> NDArray:
    """Rebuild the gradient from scratch after un-shrinking.

    Computes::

        ∇f = Q α + p

    by recomputing Q-columns for all active (non-bound) variables.

    This is called when the shrunken sub-problem appears converged but
    the full KKT check fails — i.e., some previously shrunken variables
    are back in play.

    Parameters
    ----------
    gradient : ndarray of shape (n,)   — modified in place and returned.
    alpha : ndarray of shape (n,)
    alpha_status : ndarray of int, shape (n,)
        0 = lower bound, 1 = free, 2 = upper bound.
    p, y, C : problem vectors of shape (n,).
    get_column : callable
    n : int — total number of variables.

    Returns
    -------
    gradient : ndarray of shape (n,)
    """
    # Start from p
    gradient[:] = p.copy()
    # Add Q_{:, j} * alpha_j for all j with alpha_j != 0
    for j in range(n):
        if alpha[j] == 0.0:
            continue
        col_j = get_column(j)
        gradient += alpha[j] * col_j
    return gradient


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------


def solve(
    n: int,
    Q_fn: Callable[[int, NDArray], NDArray],
    Q_diag: NDArray,
    p: NDArray,
    y: NDArray,
    lower: NDArray,
    upper: NDArray,
    *,
    tol: float = 1e-3,
    max_iter: int = 100_000,
    cache_size: int = 500,
    shrinking: bool = True,
    verbose: bool = False,
) -> SolverResult:
    """Solve the constrained QP via Sequential Minimal Optimisation.

    Minimises::

        f(α) = ½ αᵀ Q α + pᵀ α

    subject to::

        yᵀ α = 0
        lower_i ≤ α_i ≤ upper_i   for all i

    using the SMO decomposition algorithm with:

    * **Working-set selection** using second-order information (WSS-3).
    * **LRU kernel column caching** to avoid redundant evaluations.
    * **Shrinking** to skip variables that have likely reached their bounds.

    Parameters
    ----------
    n : int
        Number of dual variables (= number of training samples, or 2× for SVR).
    Q_fn : callable (i: int, indices: NDArray) -> NDArray
        Returns the *i*-th column of Q evaluated at the given row *indices*.
        Signature: ``Q_fn(i, np.arange(n)) -> ndarray of shape (n,)``.
    Q_diag : ndarray of shape (n,)
        Diagonal elements Q_ii (avoids repeated column fetches for WSS).
    p : ndarray of shape (n,)
        Linear objective coefficient vector.
    y : ndarray of shape (n,)
        Label vector (±1); encodes the equality constraint ``yᵀ α = 0``.
    lower : ndarray of shape (n,)
        Lower bounds on α (typically all zeros).
    upper : ndarray of shape (n,)
        Upper bounds on α (C for classification, C/l for scaled variants).
    tol : float, default=1e-3
        KKT violation tolerance ε for convergence.
    max_iter : int, default=100_000
        Maximum number of SMO iterations.
    cache_size : int, default=500
        Maximum number of Q-columns to hold in the LRU cache.
    shrinking : bool, default=True
        Whether to apply the shrinking heuristic.
    verbose : bool, default=False
        Print convergence info every 1000 iterations.

    Returns
    -------
    SolverResult
        Contains the optimal ``alpha``, objective ``obj``, bias ``rho``,
        iteration count ``n_iter``, and feasibility residual ``r_sq``.
    """
    # ------------------------------------------------------------------
    # Initialise
    # ------------------------------------------------------------------
    alpha = np.clip(np.zeros(n), lower, upper)  # feasible start
    gradient = p.copy()  # ∇f = Q·α + p = p  (α=0 initially)

    cache = LRUKernelCache(capacity=cache_size)

    # ---- Wrap Q_fn with caching ----
    all_indices = np.arange(n, dtype=np.int32)

    def get_column(i: int) -> NDArray:
        col = cache.get(i)
        if col is None:
            col = Q_fn(i, all_indices)
            cache.put(i, col)
        return col

    # Track which variables are active (not shrunk)
    active = np.ones(n, dtype=bool)

    # Counters for shrinking heuristic
    shrink_iter = max(2 * n, 1000)
    counter = shrink_iter

    n_iter = 0
    unshrink_done = False

    # ------------------------------------------------------------------
    # Main SMO loop
    # ------------------------------------------------------------------
    for it in range(max_iter):
        n_iter = it + 1

        # --- Periodic shrinking ---
        if shrinking:
            counter -= 1
            if counter == 0:
                counter = shrink_iter
                _shrink(alpha, gradient, y, upper, lower, active, tol)

        # --- Working set selection ---
        i, j = _select_working_set(
            gradient, y, alpha, upper, Q_diag, get_column, active
        )

        if i == -1:
            # Appears converged on active sub-problem
            if not shrinking or not active.all():
                # Unshrink: restore all variables and recheck
                if not unshrink_done:
                    unshrink_done = True
                    active[:] = True
                    _reconstruct_gradient(
                        gradient, alpha, _alpha_status(alpha, lower, upper),
                        p, y, upper, get_column, n
                    )
                    # Re-select on full problem
                    i, j = _select_working_set(
                        gradient, y, alpha, upper, Q_diag, get_column, active
                    )
                    if i == -1:
                        break   # truly converged
                else:
                    break
            else:
                break

        unshrink_done = False  # reset flag after a successful iteration

        # --- Fetch columns for i and j ---
        col_i = get_column(i)
        col_j = get_column(j)

        Q_ii = Q_diag[i]
        Q_jj = Q_diag[j]
        Q_ij = col_i[j]

        # --- Solve 2-variable sub-problem analytically ---
        a_i_old = alpha[i]
        a_j_old = alpha[j]

        # Gradient descent direction in the 2D sub-space:
        #   grad_i = y_i * ∇f_i,  grad_j = y_j * ∇f_j
        # Step:  Δ = (−y_i·∇f_i + y_j·∇f_j) / (Q_ii + Q_jj − 2·y_i·y_j·Q_ij)
        #
        # Note: for y_i == y_j, both move in the same direction;
        #       for y_i != y_j, they move in opposite directions.
        denom = Q_ii + Q_jj - 2.0 * y[i] * y[j] * Q_ij
        if denom <= 0.0:
            denom = _TAU   # numerical safety

        # Raw (unclamped) step
        delta = (-y[i] * gradient[i] + y[j] * gradient[j]) / denom

        # Apply the step
        alpha[i] += y[i] * delta
        alpha[j] -= y[j] * delta

        # Project onto feasible box [lower, upper]
        alpha[i] = float(np.clip(alpha[i], lower[i], upper[i]))
        alpha[j] = float(np.clip(alpha[j], lower[j], upper[j]))

        # Actual changes
        d_i = alpha[i] - a_i_old
        d_j = alpha[j] - a_j_old

        # --- Update gradient ---
        if d_i != 0.0 or d_j != 0.0:
            gradient += d_i * col_i + d_j * col_j

        if verbose and n_iter % 1000 == 0:
            gap = _compute_gap(gradient, y, alpha, upper, lower)
            print(f"  iter={n_iter:6d}  KKT-gap={gap:.4e}  "
                  f"active={active.sum()}/{n}")

    # ------------------------------------------------------------------
    # Compute bias (rho) from free support vectors
    # ------------------------------------------------------------------
    rho = _compute_rho(gradient, y, alpha, upper, lower)

    # ------------------------------------------------------------------
    # Compute objective value  ½ αᵀ Q α + pᵀ α
    # ------------------------------------------------------------------
    obj = 0.5 * float(alpha @ gradient) - 0.5 * float(alpha @ p) + float(p @ alpha)
    # Equivalently: obj = 0.5*(alpha @ (gradient + p))
    # Because ∇f = Q α + p  =>  Q α = gradient - p
    # So αᵀ Q α = αᵀ (∇f - p) = α·gradient - α·p
    # Objective = 0.5 α·gradient - 0.5 α·p + p·α
    #           = 0.5 α·gradient + 0.5 p·α
    obj = 0.5 * float(alpha @ (gradient + p))

    # Feasibility residual
    r_sq = float(abs(y @ alpha))

    if verbose:
        gap = _compute_gap(gradient, y, alpha, upper, lower)
        print(f"  Converged: iter={n_iter}  KKT-gap={gap:.4e}  rho={rho:.6g}")

    return SolverResult(alpha=alpha, obj=obj, rho=rho, n_iter=n_iter, r_sq=r_sq)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _alpha_status(
    alpha: NDArray, lower: NDArray, upper: NDArray
) -> NDArray:
    """Return integer status: 0=lower-bound, 1=free, 2=upper-bound."""
    status = np.ones(len(alpha), dtype=np.int8)  # default free
    status[alpha <= lower] = 0
    status[alpha >= upper] = 2
    return status


def _compute_gap(
    gradient: NDArray,
    y: NDArray,
    alpha: NDArray,
    upper: NDArray,
    lower: NDArray,
) -> float:
    """Compute the KKT optimality gap max(I_up) − min(I_low) of -y·∇f.

    A gap ≤ ε means KKT conditions are satisfied within tolerance ε.
    """
    m_yg = -y * gradient

    I_up = ((y > 0) & (alpha < upper)) | ((y < 0) & (alpha > lower))
    I_low = ((y > 0) & (alpha > lower)) | ((y < 0) & (alpha < upper))

    if not I_up.any() or not I_low.any():
        return 0.0

    return float(m_yg[I_up].max() - m_yg[I_low].min())


def _shrink(
    alpha: NDArray,
    gradient: NDArray,
    y: NDArray,
    upper: NDArray,
    lower: NDArray,
    active: NDArray,
    tol: float,
) -> None:
    """Mark variables as shrunken if they are unlikely to move.

    A variable αᵢ is shrunk if:

    * αᵢ = upper_i AND  −yᵢ ∇fᵢ ≤ min_{I_low} (−yⱼ ∇fⱼ)  (can't increase)
    * αᵢ = lower_i AND  −yᵢ ∇fᵢ ≥ max_{I_up} (−yⱼ ∇fⱼ)   (can't decrease)

    This is the heuristic from Section 5 of Chang & Lin (2011).
    Modifies *active* in place.
    """
    m_yg = -y * gradient

    I_up = ((y > 0) & (alpha < upper)) | ((y < 0) & (alpha > lower))
    I_low = ((y > 0) & (alpha > lower)) | ((y < 0) & (alpha < upper))

    if not I_up.any() or not I_low.any():
        return

    Gmax = float(m_yg[I_up].max())
    Gmin = float(m_yg[I_low].min())

    if Gmax - Gmin < tol:
        return  # already converged, don't shrink further

    # Shrink: upper-bound variables that can't possibly move upward
    at_upper = alpha >= upper
    at_lower = alpha <= lower

    shrink_upper = at_upper & (m_yg <= Gmin)
    shrink_lower = at_lower & (m_yg >= Gmax)

    active[shrink_upper | shrink_lower] = False


def _compute_rho(
    gradient: NDArray,
    y: NDArray,
    alpha: NDArray,
    upper: NDArray,
    lower: NDArray,
) -> float:
    """Compute the optimal bias ρ (decision threshold).

    For SVM the optimal b = −ρ satisfies the complementary-slackness
    conditions at free support vectors::

        ρ = (1/|free|) Σ_{i: free} y_i · ∇f_i

    At the KKT optimal point for a free variable: y_i * ∇f_i = −ρ_true
    where ρ_true is the bias in  f(x) = w·x − ρ_true.
    However the solver's rho convention is ρ = − (bias) so the formula
    is:  rho_solver = mean(-y_i * gradient_i)  for free variables.

    If no free SVs exist, we use the average of the KKT bounds.
    """
    eps = 1e-8
    free_mask = (alpha > lower + eps) & (alpha < upper - eps)

    if free_mask.any():
        # At KKT optimality for a free α_i:
        #   ∇f_i = 0  (gradient of Lagrangian w.r.t. free variable = 0)
        # which in the solver formulation means  y_i * (-y_i * ∇f_i) = ρ
        # → ρ = mean(-gradient[free] * y[free])  ... but sign depends on convention.
        # LIBSVM convention: rho is the threshold such that f(x) = sum_j alpha_j y_j K(xj,x) - rho
        # KKT for free SV: sum_j alpha_j y_j K(xj, xi) = rho + y_i * (something)  
        # More precisely: at free α_i, ∂L/∂α_i = 0 → (Qα)_i + p_i = ν*y_i  for some ν
        # where ν is the dual of the equality constraint. For C-SVC, p_i = -1:
        # (Qα)_i - 1 = ν*y_i → (Qα)_i = 1 + ν*y_i
        # and rho = -ν (the KKT multiplier for yᵀα=0).
        # In the solver: gradient[i] = (Qα + p)[i], so:
        # gradient[i] = 1 + ν*y_i - 1 = ν*y_i  at free SVs for C-SVC  
        # Wait: gradient[i] = Σ_j Q_ij α_j + p_i  and at KKT for free α_i:
        # gradient[i] * y_i = (Qα)_i * y_i + p_i * y_i
        # The KKT condition for the primal is: y_i f(xi) = 1 for free SVs
        # Decision: f(xi) = Σ_j α_j y_j K(xj,xi) - rho
        # = (1/y_i) * [(Qα)_i / 1] - rho  (since Q_ij = y_i y_j K(xi,xj))
        # = gradient[i]/y_i - p_i/y_i - rho  (using gradient = Qα + p)
        # At KKT for free SV of C-SVC: f(xi) = 1/y_i (sign of margin)
        # Hmm, let's just use the LIBSVM svm.cpp approach directly:
        # rho = (Gmax + Gmin) / 2  where Gmax = max_{I_up} -y*grad, Gmin = min_{I_low} -y*grad
        # But preferably average over free SVs: rho = mean(-gradient[free]*y[free])? 
        # No: in svm.cpp the bias calculation is:
        # For free SVs: rho += -y[i]*G[i]  (G = gradient, y = ±1)
        # So rho = mean(-y * gradient) over free SVs.
        return float(np.mean(-y[free_mask] * gradient[free_mask]))

    # Fall back: average of KKT upper and lower bounds
    m_yg = -y * gradient
    I_up = ((y > 0) & (alpha < upper)) | ((y < 0) & (alpha > lower))
    I_low = ((y > 0) & (alpha > lower)) | ((y < 0) & (alpha < upper))

    ub = m_yg[I_up].max() if I_up.any() else 0.0
    lb = m_yg[I_low].min() if I_low.any() else 0.0
    return float((ub + lb) / 2.0)
