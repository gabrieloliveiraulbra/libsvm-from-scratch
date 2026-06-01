"""
antigravity.svm.multiclass
==========================
One-Against-One (OAO) multiclass strategy for SVM classification.

For *k* classes the strategy trains ``k(k−1)/2`` binary classifiers and
uses a **voting** scheme to determine the final prediction — each
classifier casts one vote for its winning class, and the class with the
most votes wins (ties broken by lowest class index).

This is the exact approach described in Chang & Lin (2011), Section 7,
and is the default multiclass method in LIBSVM.

Reference
---------
Chang, C.-C., & Lin, C.-J. (2011).
    LIBSVM: A library for support vector machines. ACM TIST, 2(3), 27.
Hsu, C.-W., & Lin, C.-J. (2002).
    A comparison of methods for multiclass support vector machines.
    IEEE Transactions on Neural Networks, 13(2), 415–425.
"""

from __future__ import annotations

from itertools import combinations
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from antigravity.svm.models import CSVC, _default_gamma
from antigravity.svm.kernels import KERNEL_RBF


class MulticlassSVC:
    """One-Against-One multiclass Support Vector Classifier.

    Trains ``k(k−1)/2`` binary :class:`~antigravity.svm.models.CSVC`
    classifiers — one for each pair of distinct classes — then aggregates
    predictions via majority voting.

    Parameters
    ----------
    C : float, default=1.0
        Regularisation parameter passed to each binary classifier.
    kernel : str, default='rbf'
    gamma : float or None
        Kernel coefficient (None → 1/n_features).
    coef0 : float, default=0.0
    degree : int, default=3
    tol : float, default=1e-3
    max_iter : int, default=-1
    cache_size : int, default=500
    shrinking : bool, default=True
    verbose : bool, default=False

    Attributes
    ----------
    classes_ : ndarray of shape (k,)
        Unique class labels discovered during ``fit``.
    classifiers_ : list of CSVC
        The ``k(k−1)/2`` fitted binary classifiers.
    pairs_ : list of tuple (i, j)
        Each entry ``(i, j)`` indicates which pair of *class-index* values
        (into ``classes_``) the corresponding classifier handles.
    n_classes_ : int
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
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.tol = tol
        self.max_iter = max_iter
        self.cache_size = cache_size
        self.shrinking = shrinking
        self.verbose = verbose

        self.classes_: Optional[NDArray] = None
        self.classifiers_: list[CSVC] = []
        self.pairs_: list[tuple[int, int]] = []
        self.n_classes_: int = 0

    # ------------------------------------------------------------------
    def fit(self, X: NDArray, y: NDArray) -> "MulticlassSVC":
        """Train ``k(k−1)/2`` binary classifiers.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            Class labels.  Must contain at least 2 distinct values.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        if self.n_classes_ < 2:
            raise ValueError("MulticlassSVC requires at least 2 classes.")

        self.classifiers_ = []
        self.pairs_ = []

        class_pairs = list(combinations(range(self.n_classes_), 2))

        for ci, cj in class_pairs:
            label_i = self.classes_[ci]
            label_j = self.classes_[cj]

            # Select samples belonging to either class
            mask = (y == label_i) | (y == label_j)
            X_sub = X[mask]
            y_sub = y[mask]

            clf = CSVC(
                C=self.C, kernel=self.kernel, gamma=self.gamma,
                coef0=self.coef0, degree=self.degree, tol=self.tol,
                max_iter=self.max_iter, cache_size=self.cache_size,
                shrinking=self.shrinking, verbose=self.verbose,
            )
            clf.fit(X_sub, y_sub)
            self.classifiers_.append(clf)
            self.pairs_.append((ci, cj))

        return self

    # ------------------------------------------------------------------
    def decision_function(self, X: NDArray) -> NDArray:
        """Compute vote counts for each class.

        Returns
        -------
        votes : ndarray of shape (n_samples, n_classes)
            ``votes[s, c]`` is the number of pairwise classifiers that
            voted for class ``c`` for sample ``s``.
        """
        if not self.classifiers_:
            raise RuntimeError("Call .fit() first.")

        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        votes = np.zeros((n_samples, self.n_classes_), dtype=np.float64)

        for clf, (ci, cj) in zip(self.classifiers_, self.pairs_):
            decisions = clf.decision_function(X)  # shape (n_samples,)
            # clf.classes_[1] is the 'positive' class (mapped to +1 inside CSVC)
            # clf.classes_[0] is the 'negative' class (mapped to -1 inside CSVC)
            pos_label = clf.classes_[1]
            neg_label = clf.classes_[0]
            # Find which global class index corresponds to each binary label
            pos_global = ci if self.classes_[ci] == pos_label else cj
            neg_global = cj if pos_global == ci else ci
            for s in range(n_samples):
                if decisions[s] >= 0:
                    votes[s, pos_global] += 1
                else:
                    votes[s, neg_global] += 1

        return votes

    # ------------------------------------------------------------------
    def predict(self, X: NDArray) -> NDArray:
        """Predict class labels for X using majority voting.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels in the original label space.
        """
        votes = self.decision_function(X)
        # Ties: argmax returns the first (lowest index) winner — deterministic
        winner_idx = np.argmax(votes, axis=1)
        return self.classes_[winner_idx]

    # ------------------------------------------------------------------
    def predict_proba(self, X: NDArray) -> NDArray:
        """Estimate class probabilities via vote-count normalisation.

        .. note::
            This is a coarse approximation — use
            :class:`~antigravity.svm.probability.PlattScaling` wrapped
            around each binary classifier for calibrated probabilities.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Each row sums to 1.
        """
        votes = self.decision_function(X)
        totals = votes.sum(axis=1, keepdims=True)
        totals = np.where(totals == 0, 1, totals)  # avoid div-by-zero
        return votes / totals
