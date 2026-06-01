"""
antigravity.svm.tuning
======================
Grid-search cross-validation hyperparameter tuner for SVM models.

:class:`GridSearchCV` exhaustively evaluates all combinations of
parameters in a supplied grid and selects the combination with the
best cross-validated score (accuracy for classifiers, negative-MSE for
regressors).

The implementation uses a *stratified k-fold* split for classifiers and
a *random k-fold* split for regressors.  Parallel evaluation across
parameter configurations is supported via
:mod:`concurrent.futures.ProcessPoolExecutor`.

Example
-------
>>> from antigravity.svm import CSVC, GridSearchCV
>>> param_grid = {"C": [0.1, 1, 10], "gamma": [0.01, 0.1, 1]}
>>> search = GridSearchCV(CSVC, param_grid, cv=5, scoring="accuracy")
>>> search.fit(X_train, y_train)
>>> print(search.best_params_, search.best_score_)
"""

from __future__ import annotations

import math
import itertools
import concurrent.futures
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Cross-validation split helpers
# ---------------------------------------------------------------------------


def _stratified_kfold(
    y: NDArray, n_splits: int
) -> List[Tuple[NDArray, NDArray]]:
    """Stratified k-fold split preserving class ratios in each fold.

    Parameters
    ----------
    y : ndarray of shape (n,) — class labels.
    n_splits : int — number of folds.

    Returns
    -------
    splits : list of (train_indices, val_indices) tuples.
    """
    classes, y_idx = np.unique(y, return_inverse=True)
    n = len(y)
    fold_indices = [[] for _ in range(n_splits)]

    for c in range(len(classes)):
        class_idx = np.where(y_idx == c)[0]
        np.random.shuffle(class_idx)
        for fold_i, part in enumerate(np.array_split(class_idx, n_splits)):
            fold_indices[fold_i].append(part)

    splits = []
    for fold_i in range(n_splits):
        val_idx = np.concatenate(fold_indices[fold_i])
        train_idx = np.concatenate(
            [np.concatenate(fold_indices[j])
             for j in range(n_splits) if j != fold_i]
        )
        splits.append((train_idx.astype(np.int64), val_idx.astype(np.int64)))

    return splits


def _kfold(n: int, n_splits: int) -> List[Tuple[NDArray, NDArray]]:
    """Standard k-fold split for regression tasks.

    Parameters
    ----------
    n : int — number of samples.
    n_splits : int.

    Returns
    -------
    splits : list of (train_indices, val_indices) tuples.
    """
    indices = np.arange(n)
    np.random.shuffle(indices)
    splits = []
    for val_idx in np.array_split(indices, n_splits):
        train_idx = np.setdiff1d(indices, val_idx)
        splits.append((train_idx, val_idx))
    return splits


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------


def _score_accuracy(y_true: NDArray, y_pred: NDArray) -> float:
    return float(np.mean(y_true == y_pred))


def _score_neg_mse(y_true: NDArray, y_pred: NDArray) -> float:
    return -float(np.mean((y_true - y_pred) ** 2))


_SCORING_MAP: Dict[str, Callable] = {
    "accuracy": _score_accuracy,
    "neg_mse": _score_neg_mse,
}


# ---------------------------------------------------------------------------
# Internal worker (must be module-level for ProcessPoolExecutor pickling)
# ---------------------------------------------------------------------------


def _evaluate_params(
    model_cls,
    model_kwargs: Dict[str, Any],
    X: NDArray,
    y: NDArray,
    splits: List[Tuple[NDArray, NDArray]],
    scoring_fn: Callable,
) -> Tuple[Dict[str, Any], float, List[float]]:
    """Evaluate a single parameter configuration via cross-validation.

    Returns
    -------
    (params, mean_score, fold_scores)
    """
    fold_scores = []
    for train_idx, val_idx in splits:
        model = model_cls(**model_kwargs)
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[val_idx])
        fold_scores.append(scoring_fn(y[val_idx], y_pred))

    mean_score = float(np.mean(fold_scores))
    return model_kwargs, mean_score, fold_scores


# ---------------------------------------------------------------------------
# GridSearchCV
# ---------------------------------------------------------------------------


class GridSearchCV:
    """Exhaustive grid-search with cross-validation.

    Parameters
    ----------
    model_cls : class
        Uninstantiated SVM model class (e.g., :class:`~antigravity.svm.CSVC`).
    param_grid : dict
        Dictionary mapping parameter names to lists of values to try.
        Example: ``{"C": [0.1, 1, 10], "gamma": [0.01, 0.1]}``.
    cv : int, default=5
        Number of cross-validation folds.
    scoring : {'accuracy', 'neg_mse'} or callable, default='accuracy'
        Metric to optimise.  Use ``'accuracy'`` for classifiers and
        ``'neg_mse'`` for regressors.  A custom callable must have the
        signature ``f(y_true, y_pred) -> float`` (higher is better).
    n_jobs : int, default=1
        Number of parallel workers.  Set to ``-1`` to use all available CPUs.
    random_state : int or None, default=None
        Seed for reproducible fold splits.
    fixed_params : dict or None, default=None
        Parameters always passed to the model constructor (not searched).

    Attributes
    ----------
    best_params_ : dict
        Parameter combination that achieved the highest CV score.
    best_score_ : float
        Cross-validated score of the best configuration.
    cv_results_ : list of dict
        One entry per parameter configuration, with keys
        ``'params'``, ``'mean_score'``, ``'fold_scores'``.
    best_estimator_ : fitted model
        Model re-fitted on the full training set with ``best_params_``.
    """

    def __init__(
        self,
        model_cls,
        param_grid: Dict[str, List[Any]],
        cv: int = 5,
        scoring: str | Callable = "accuracy",
        n_jobs: int = 1,
        random_state: Optional[int] = None,
        fixed_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_cls = model_cls
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.fixed_params = fixed_params or {}

        self.best_params_: Dict[str, Any] = {}
        self.best_score_: float = -math.inf
        self.cv_results_: List[Dict[str, Any]] = []
        self.best_estimator_ = None

    # ------------------------------------------------------------------
    def _get_scoring_fn(self) -> Callable:
        if callable(self.scoring):
            return self.scoring
        if self.scoring not in _SCORING_MAP:
            raise ValueError(
                f"Unknown scoring '{self.scoring}'. "
                f"Choose from {list(_SCORING_MAP)} or supply a callable."
            )
        return _SCORING_MAP[self.scoring]

    # ------------------------------------------------------------------
    def _build_param_combinations(self) -> List[Dict[str, Any]]:
        """Expand the param_grid into a flat list of dicts."""
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combos = []
        for combo in itertools.product(*values):
            params = {**self.fixed_params, **dict(zip(keys, combo))}
            combos.append(params)
        return combos

    # ------------------------------------------------------------------
    def fit(self, X: NDArray, y: NDArray) -> "GridSearchCV":
        """Run grid search and fit the best model on the full training set.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,)

        Returns
        -------
        self
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        scoring_fn = self._get_scoring_fn()
        combos = self._build_param_combinations()

        # Determine fold strategy
        is_classification = (self.scoring == "accuracy" or
                             (callable(self.scoring) and
                              getattr(self.scoring, "_is_classification", False)))

        if is_classification:
            splits = _stratified_kfold(y, self.cv)
        else:
            splits = _kfold(len(y), self.cv)

        self.cv_results_ = []
        results: List[Tuple[Dict, float, List[float]]] = []

        n_jobs = self.n_jobs if self.n_jobs > 0 else None   # None = all CPUs

        if n_jobs == 1:
            # Sequential evaluation
            for params in combos:
                r = _evaluate_params(
                    self.model_cls, params, X, y, splits, scoring_fn
                )
                results.append(r)
        else:
            # Parallel evaluation
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=n_jobs
            ) as executor:
                futures = {
                    executor.submit(
                        _evaluate_params,
                        self.model_cls, params, X, y, splits, scoring_fn,
                    ): params
                    for params in combos
                }
                for fut in concurrent.futures.as_completed(futures):
                    results.append(fut.result())

        # Collect results
        best_score = -math.inf
        best_params: Dict[str, Any] = {}
        for params, mean_score, fold_scores in results:
            self.cv_results_.append({
                "params": params,
                "mean_score": mean_score,
                "fold_scores": fold_scores,
            })
            if mean_score > best_score:
                best_score = mean_score
                best_params = params

        self.best_score_ = best_score
        self.best_params_ = {
            k: v for k, v in best_params.items() if k not in self.fixed_params
        }

        # Refit best model on full data
        all_params = {**self.fixed_params, **self.best_params_}
        self.best_estimator_ = self.model_cls(**all_params)
        self.best_estimator_.fit(X, y)

        return self

    # ------------------------------------------------------------------
    def predict(self, X: NDArray) -> NDArray:
        """Predict using the best estimator.

        Parameters
        ----------
        X : ndarray of shape (m, n_features)

        Returns
        -------
        y_pred : ndarray of shape (m,)
        """
        if self.best_estimator_ is None:
            raise RuntimeError("Call .fit() first.")
        return self.best_estimator_.predict(X)

    # ------------------------------------------------------------------
    def summary(self) -> str:
        """Return a formatted summary of CV results sorted by score.

        Returns
        -------
        str
        """
        if not self.cv_results_:
            return "No results yet. Call .fit() first."

        rows = sorted(self.cv_results_, key=lambda r: r["mean_score"], reverse=True)
        lines = [
            f"{'Rank':<5} {'Score':>10}  {'Params'}",
            "-" * 60,
        ]
        for rank, row in enumerate(rows, 1):
            param_str = ", ".join(f"{k}={v}" for k, v in row["params"].items()
                                  if k not in self.fixed_params)
            lines.append(f"{rank:<5} {row['mean_score']:>10.4f}  {param_str}")

        lines.append("-" * 60)
        lines.append(
            f"Best: score={self.best_score_:.4f}  "
            + ", ".join(f"{k}={v}" for k, v in self.best_params_.items())
        )
        return "\n".join(lines)
