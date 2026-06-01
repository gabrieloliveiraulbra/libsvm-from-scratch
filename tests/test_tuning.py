"""Tests for antigravity.svm.tuning"""
import numpy as np
import pytest
from antigravity.svm.tuning import GridSearchCV, _stratified_kfold, _kfold
from antigravity.svm.models import CSVC, EpsilonSVR
from antigravity.svm.kernels import KERNEL_LINEAR, KERNEL_RBF


# ---------------------------------------------------------------------------
# Fold helpers
# ---------------------------------------------------------------------------


class TestFoldHelpers:
    def test_stratified_kfold_sizes(self):
        rng = np.random.RandomState(0)
        y = np.array([0] * 50 + [1] * 50)
        splits = _stratified_kfold(y, n_splits=5)
        assert len(splits) == 5
        for train, val in splits:
            assert len(train) + len(val) == 100

    def test_stratified_kfold_no_overlap(self):
        y = np.array([0] * 30 + [1] * 30)
        splits = _stratified_kfold(y, n_splits=3)
        for train, val in splits:
            assert len(set(train) & set(val)) == 0

    def test_kfold_sizes(self):
        splits = _kfold(n=90, n_splits=5)
        assert len(splits) == 5
        for train, val in splits:
            assert len(train) + len(val) == 90

    def test_kfold_no_overlap(self):
        splits = _kfold(n=60, n_splits=4)
        for train, val in splits:
            assert len(np.intersect1d(train, val)) == 0


# ---------------------------------------------------------------------------
# GridSearchCV — classification
# ---------------------------------------------------------------------------


@pytest.fixture
def separable_dataset():
    rng = np.random.RandomState(0)
    X_pos = rng.randn(30, 2) + np.array([3.0, 3.0])
    X_neg = rng.randn(30, 2) + np.array([-3.0, -3.0])
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * 30 + [-1] * 30)
    return X, y


class TestGridSearchCVClassification:
    def test_fit_finds_best_params(self, separable_dataset):
        X, y = separable_dataset
        param_grid = {"C": [0.1, 1.0, 10.0]}
        gs = GridSearchCV(
            CSVC, param_grid, cv=3, scoring="accuracy",
            random_state=0,
            fixed_params={"kernel": KERNEL_LINEAR},
        )
        gs.fit(X, y)
        assert gs.best_score_ > 0.90
        assert "C" in gs.best_params_

    def test_cv_results_length(self, separable_dataset):
        X, y = separable_dataset
        param_grid = {"C": [0.1, 1.0], "gamma": [0.1, 1.0]}
        gs = GridSearchCV(
            CSVC, param_grid, cv=3, scoring="accuracy",
            random_state=42,
            fixed_params={"kernel": KERNEL_RBF},
        )
        gs.fit(X, y)
        # 2 × 2 = 4 combinations
        assert len(gs.cv_results_) == 4

    def test_best_estimator_fitted(self, separable_dataset):
        X, y = separable_dataset
        gs = GridSearchCV(
            CSVC, {"C": [1.0, 10.0]}, cv=3, scoring="accuracy",
            random_state=1,
            fixed_params={"kernel": KERNEL_LINEAR},
        )
        gs.fit(X, y)
        assert gs.best_estimator_ is not None
        y_pred = gs.predict(X)
        assert y_pred.shape == (len(y),)

    def test_predict_before_fit_raises(self, separable_dataset):
        X, _ = separable_dataset
        gs = GridSearchCV(CSVC, {"C": [1.0]})
        with pytest.raises(RuntimeError):
            gs.predict(X)

    def test_best_params_in_grid(self, separable_dataset):
        X, y = separable_dataset
        C_values = [0.01, 0.1, 1.0, 10.0]
        gs = GridSearchCV(
            CSVC, {"C": C_values}, cv=3, scoring="accuracy",
            random_state=5,
            fixed_params={"kernel": KERNEL_LINEAR},
        )
        gs.fit(X, y)
        assert gs.best_params_["C"] in C_values

    def test_summary_returns_string(self, separable_dataset):
        X, y = separable_dataset
        gs = GridSearchCV(
            CSVC, {"C": [1.0, 10.0]}, cv=3, scoring="accuracy",
            random_state=0,
            fixed_params={"kernel": KERNEL_LINEAR},
        )
        gs.fit(X, y)
        summary = gs.summary()
        assert isinstance(summary, str)
        assert "Best" in summary


# ---------------------------------------------------------------------------
# GridSearchCV — regression
# ---------------------------------------------------------------------------


@pytest.fixture
def regression_dataset():
    rng = np.random.RandomState(7)
    X = rng.randn(40, 2)
    y = 3.0 * X[:, 0] - X[:, 1] + 0.5 * rng.randn(40)
    return X, y


class TestGridSearchCVRegression:
    def test_fit_neg_mse(self, regression_dataset):
        X, y = regression_dataset
        gs = GridSearchCV(
            EpsilonSVR,
            {"C": [1.0, 10.0], "epsilon": [0.05, 0.1]},
            cv=3,
            scoring="neg_mse",
            random_state=0,
            fixed_params={"kernel": KERNEL_LINEAR},
        )
        gs.fit(X, y)
        # neg_mse should be < 0
        assert gs.best_score_ < 0
        assert gs.best_estimator_ is not None

    def test_custom_scoring(self, regression_dataset):
        X, y = regression_dataset

        def mae_score(y_true, y_pred):
            return -float(np.mean(np.abs(y_true - y_pred)))

        gs = GridSearchCV(
            EpsilonSVR,
            {"C": [1.0, 10.0]},
            cv=3,
            scoring=mae_score,
            random_state=0,
            fixed_params={"kernel": KERNEL_LINEAR, "epsilon": 0.1},
        )
        gs.fit(X, y)
        assert gs.best_estimator_ is not None

    def test_unknown_scoring_raises(self, regression_dataset):
        X, y = regression_dataset
        gs = GridSearchCV(EpsilonSVR, {"C": [1.0]}, scoring="f1_score")
        with pytest.raises(ValueError, match="Unknown scoring"):
            gs.fit(X, y)
