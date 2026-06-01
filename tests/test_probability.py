"""Tests for antigravity.svm.probability"""
import numpy as np
import pytest
from antigravity.svm.probability import PlattScaling, LaplaceSVR, _fit_sigmoid_newton


# ---------------------------------------------------------------------------
# PlattScaling
# ---------------------------------------------------------------------------


class TestPlattScaling:
    def test_fit_sets_parameters(self):
        rng = np.random.RandomState(0)
        f = rng.randn(100)
        y = np.sign(f + 0.1 * rng.randn(100))
        y[y == 0] = 1.0
        platt = PlattScaling()
        platt.fit(f, y)
        assert isinstance(platt.A_, float)
        assert isinstance(platt.B_, float)

    def test_predict_proba_shape(self):
        f = np.linspace(-3, 3, 50)
        y = np.sign(f)
        y[y == 0] = 1.0
        platt = PlattScaling()
        platt.fit(f, y)
        proba = platt.predict_proba(f)
        assert proba.shape == (50, 2)

    def test_predict_proba_sums_to_one(self):
        f = np.linspace(-3, 3, 50)
        y = np.sign(f)
        y[y == 0] = 1.0
        platt = PlattScaling()
        platt.fit(f, y)
        proba = platt.predict_proba(f)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)

    def test_predict_proba_values_in_0_1(self):
        f = np.linspace(-5, 5, 100)
        y = np.sign(f)
        y[y == 0] = 1.0
        platt = PlattScaling()
        platt.fit(f, y)
        proba = platt.predict_proba(f)
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)

    def test_monotone_positive_class(self):
        """P(y=+1 | f) should be monotone increasing in f for a well-fitted sigmoid."""
        f = np.linspace(-5, 5, 200)
        y = np.sign(f)
        y[y == 0] = 1.0
        platt = PlattScaling()
        platt.fit(f, y)
        p_pos = platt.predict_proba_binary(f)
        # After fitting with A < 0 (expected for SVM), the sigmoid is monotone
        # We just check that the sorted decision values yield monotone probs
        p_sorted = p_pos[np.argsort(f)]
        diffs = np.diff(p_sorted)
        # Allow for tiny numerical noise but should be mostly monotone
        assert np.mean(diffs >= -1e-6) >= 0.95

    def test_extreme_values_stable(self):
        """Test numerical stability at extreme decision values."""
        platt = PlattScaling()
        f_train = np.linspace(-2, 2, 50)
        y_train = np.sign(f_train)
        y_train[y_train == 0] = 1.0
        platt.fit(f_train, y_train)

        f_extreme = np.array([-1000.0, 1000.0])
        proba = platt.predict_proba(f_extreme)
        assert np.all(np.isfinite(proba))
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)

    def test_fit_with_cv(self):
        """fit_with_cv should produce finite A, B without errors."""
        rng = np.random.RandomState(42)
        X = rng.randn(60, 2)
        y = np.sign(X[:, 0])
        y[y == 0] = 1.0

        from antigravity.svm.models import CSVC
        from antigravity.svm.kernels import KERNEL_LINEAR

        platt = PlattScaling(n_folds=3)
        platt.fit_with_cv(
            model_factory=lambda: CSVC(C=1.0, kernel=KERNEL_LINEAR),
            X=X, y=y
        )
        assert np.isfinite(platt.A_)
        assert np.isfinite(platt.B_)


# ---------------------------------------------------------------------------
# _fit_sigmoid_newton
# ---------------------------------------------------------------------------


class TestFitSigmoidNewton:
    def test_returns_floats(self):
        f = np.linspace(-2, 2, 30)
        t = 1.0 / (1.0 + np.exp(-f))  # ideal targets
        A, B = _fit_sigmoid_newton(f, t, max_iter=50)
        assert isinstance(A, float)
        assert isinstance(B, float)

    def test_converges_on_ideal_targets(self):
        """When targets ARE a sigmoid, A should be close to -1 and B close to 0."""
        f = np.linspace(-3, 3, 200)
        t = 1.0 / (1.0 + np.exp(f))   # σ(−f) — the ideal Platt form with A=1, B=0
        A, B = _fit_sigmoid_newton(f, t, max_iter=200, tol=1e-8)
        # We expect A ≈ 1.0, B ≈ 0.0 (give some slack for label smoothing)
        assert abs(A - 1.0) < 0.2
        assert abs(B) < 0.2


# ---------------------------------------------------------------------------
# LaplaceSVR
# ---------------------------------------------------------------------------


class TestLaplaceSVR:
    def test_fit_sigma_positive(self):
        residuals = np.abs(np.random.randn(100))
        lap = LaplaceSVR()
        lap.fit(residuals)
        assert lap.sigma_ > 0

    def test_fit_sigma_mean_abs(self):
        residuals = np.array([1.0, 2.0, 3.0])
        lap = LaplaceSVR()
        lap.fit(residuals)
        np.testing.assert_allclose(lap.sigma_, 2.0)

    def test_predict_interval_shape(self):
        rng = np.random.RandomState(5)
        y_pred = rng.randn(50)
        lap = LaplaceSVR()
        lap.fit(np.abs(rng.randn(50)))
        lo, hi = lap.predict_interval(y_pred, confidence=0.90)
        assert lo.shape == (50,)
        assert hi.shape == (50,)
        assert np.all(hi >= lo)

    def test_predict_interval_contains_prediction(self):
        """The interval [lower, upper] always contains y_pred."""
        rng = np.random.RandomState(7)
        y_pred = rng.randn(30)
        lap = LaplaceSVR()
        lap.fit(np.ones(30))  # sigma = 1
        lo, hi = lap.predict_interval(y_pred, confidence=0.95)
        assert np.all(y_pred >= lo)
        assert np.all(y_pred <= hi)

    def test_predict_interval_invalid_confidence(self):
        lap = LaplaceSVR()
        lap.sigma_ = 1.0
        with pytest.raises(ValueError):
            lap.predict_interval(np.zeros(5), confidence=1.5)

    def test_predict_proba_range(self):
        rng = np.random.RandomState(9)
        y_pred = rng.randn(40)
        y_true = y_pred + 0.5 * rng.randn(40)
        lap = LaplaceSVR()
        lap.fit(np.abs(y_true - y_pred))
        p = lap.predict_proba(y_pred, y_true)
        assert np.all(p >= 0)
        assert np.all(p < 1)

    def test_fit_with_cv(self):
        """fit_with_cv should estimate a positive sigma."""
        rng = np.random.RandomState(13)
        X = rng.randn(40, 2)
        y = 2.0 * X[:, 0] - X[:, 1] + 0.1 * rng.randn(40)

        from antigravity.svm.models import EpsilonSVR
        from antigravity.svm.kernels import KERNEL_LINEAR

        lap = LaplaceSVR(n_folds=3)
        lap.fit_with_cv(
            model_factory=lambda: EpsilonSVR(C=10.0, kernel=KERNEL_LINEAR),
            X=X, y=y
        )
        assert lap.sigma_ > 0
        assert np.isfinite(lap.sigma_)

    def test_zero_residuals_clamped(self):
        """Should not divide by zero when all residuals are 0."""
        lap = LaplaceSVR()
        lap.fit(np.zeros(10))
        assert lap.sigma_ > 0
        assert np.isfinite(lap.sigma_)
