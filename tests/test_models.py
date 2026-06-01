"""Tests for antigravity.svm.models"""
import numpy as np
import pytest
from antigravity.svm.models import CSVC, NuSVC, OneClassSVM, EpsilonSVR, NuSVR
from antigravity.svm.kernels import KERNEL_RBF, KERNEL_LINEAR


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def binary_linear():
    """Linearly separable binary classification dataset."""
    rng = np.random.RandomState(0)
    X_pos = rng.randn(20, 2) + np.array([2.0, 2.0])
    X_neg = rng.randn(20, 2) + np.array([-2.0, -2.0])
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * 20 + [-1] * 20)
    return X, y


@pytest.fixture
def regression_linear():
    """Simple linear regression dataset."""
    rng = np.random.RandomState(1)
    X = rng.randn(30, 2)
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 0.1 * rng.randn(30)
    return X, y


# ---------------------------------------------------------------------------
# CSVC tests
# ---------------------------------------------------------------------------


class TestCSVC:
    def test_fit_predict_binary(self, binary_linear):
        X, y = binary_linear
        clf = CSVC(C=1.0, kernel=KERNEL_LINEAR, tol=1e-4)
        clf.fit(X, y)
        y_pred = clf.predict(X)
        accuracy = np.mean(y_pred == y)
        assert accuracy >= 0.95

    def test_support_vectors_set(self, binary_linear):
        X, y = binary_linear
        clf = CSVC(C=1.0, kernel=KERNEL_RBF, gamma=0.5)
        clf.fit(X, y)
        assert clf.support_vectors_ is not None
        assert clf.support_vectors_.shape[1] == X.shape[1]
        assert len(clf.support_) > 0

    def test_dual_coef_shape(self, binary_linear):
        X, y = binary_linear
        clf = CSVC(C=1.0, kernel=KERNEL_RBF, gamma=0.5)
        clf.fit(X, y)
        n_sv = len(clf.support_)
        assert clf.dual_coef_.shape == (1, n_sv)

    def test_decision_function_sign(self, binary_linear):
        X, y = binary_linear
        clf = CSVC(C=1.0, kernel=KERNEL_LINEAR)
        clf.fit(X, y)
        df = clf.decision_function(X)
        # Decision values should agree in sign with predictions
        y_pred = clf.predict(X)
        y_pred_mapped = np.where(y_pred == clf.classes_[1], 1, -1)
        sign_agreement = np.sign(df) == np.sign(y_pred_mapped)
        assert sign_agreement.all()

    def test_multiclass_raises(self):
        X = np.random.randn(10, 2)
        y = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
        clf = CSVC()
        with pytest.raises(ValueError, match="2 classes"):
            clf.fit(X, y)

    def test_not_fitted_raises(self):
        clf = CSVC()
        with pytest.raises(RuntimeError):
            clf.decision_function(np.ones((3, 2)))

    def test_different_label_types(self, binary_linear):
        X, _ = binary_linear
        y_str = np.array(["cat"] * 20 + ["dog"] * 20)
        clf = CSVC(C=1.0, kernel=KERNEL_LINEAR)
        clf.fit(X, y_str)
        y_pred = clf.predict(X)
        assert set(y_pred).issubset({"cat", "dog"})

    def test_rbf_high_accuracy(self, binary_linear):
        X, y = binary_linear
        clf = CSVC(C=10.0, kernel=KERNEL_RBF, gamma=0.5)
        clf.fit(X, y)
        assert np.mean(clf.predict(X) == y) >= 0.95


# ---------------------------------------------------------------------------
# NuSVC tests
# ---------------------------------------------------------------------------


class TestNuSVC:
    def test_fit_predict(self, binary_linear):
        X, y = binary_linear
        clf = NuSVC(nu=0.3, kernel=KERNEL_RBF, gamma=0.5)
        clf.fit(X, y)
        y_pred = clf.predict(X)
        assert np.mean(y_pred == y) >= 0.85

    def test_classes_set(self, binary_linear):
        X, y = binary_linear
        clf = NuSVC(nu=0.2)
        clf.fit(X, y)
        assert set(clf.classes_) == {-1, 1}

    def test_multiclass_raises(self):
        X = np.random.randn(9, 2)
        y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        with pytest.raises(ValueError):
            NuSVC().fit(X, y)


# ---------------------------------------------------------------------------
# OneClassSVM tests
# ---------------------------------------------------------------------------


class TestOneClassSVM:
    def test_fit_predict_inliers(self):
        """Inliers from the training distribution should be classified +1."""
        rng = np.random.RandomState(5)
        X_train = rng.randn(50, 2)
        X_test_in = rng.randn(20, 2)         # same distribution
        X_test_out = rng.randn(20, 2) + 10   # far away

        clf = OneClassSVM(nu=0.1, kernel=KERNEL_RBF, gamma=0.5)
        clf.fit(X_train)

        preds_in = clf.predict(X_test_in)
        preds_out = clf.predict(X_test_out)

        # Most inliers should be +1
        assert np.mean(preds_in == 1) >= 0.5
        # Most outliers should be -1
        assert np.mean(preds_out == -1) >= 0.7

    def test_predict_values(self):
        rng = np.random.RandomState(3)
        X = rng.randn(30, 2)
        clf = OneClassSVM(nu=0.2, kernel=KERNEL_RBF, gamma=1.0)
        clf.fit(X)
        preds = clf.predict(X)
        assert set(preds).issubset({-1, 1})

    def test_y_ignored(self):
        X = np.random.randn(20, 2)
        clf = OneClassSVM(nu=0.1)
        # Passing y should not raise
        clf.fit(X, y=np.ones(20))


# ---------------------------------------------------------------------------
# EpsilonSVR tests
# ---------------------------------------------------------------------------


class TestEpsilonSVR:
    def test_fit_predict_linear(self, regression_linear):
        X, y = regression_linear
        reg = EpsilonSVR(C=10.0, epsilon=0.1, kernel=KERNEL_LINEAR)
        reg.fit(X, y)
        y_pred = reg.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        # Should have low training error on this simple linear problem
        assert mse < 5.0

    def test_support_vectors(self, regression_linear):
        X, y = regression_linear
        reg = EpsilonSVR(C=1.0, epsilon=0.1, kernel=KERNEL_RBF, gamma=0.5)
        reg.fit(X, y)
        assert reg.support_vectors_ is not None
        assert reg.support_vectors_.shape[1] == X.shape[1]

    def test_predict_shape(self, regression_linear):
        X, y = regression_linear
        reg = EpsilonSVR(C=1.0)
        reg.fit(X, y)
        assert reg.predict(X).shape == (len(y),)

    def test_decision_function_equals_predict(self, regression_linear):
        X, y = regression_linear
        reg = EpsilonSVR(C=1.0)
        reg.fit(X, y)
        np.testing.assert_array_equal(reg.predict(X), reg.decision_function(X))


# ---------------------------------------------------------------------------
# NuSVR tests
# ---------------------------------------------------------------------------


class TestNuSVR:
    def test_fit_predict(self, regression_linear):
        X, y = regression_linear
        reg = NuSVR(C=10.0, nu=0.5, kernel=KERNEL_LINEAR)
        reg.fit(X, y)
        y_pred = reg.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        assert mse < 10.0

    def test_predict_shape(self, regression_linear):
        X, y = regression_linear
        reg = NuSVR(C=1.0, nu=0.3)
        reg.fit(X, y)
        assert reg.predict(X).shape == (len(y),)

    def test_support_vectors_shape(self, regression_linear):
        X, y = regression_linear
        reg = NuSVR(C=1.0, nu=0.5, kernel=KERNEL_RBF, gamma=0.5)
        reg.fit(X, y)
        assert reg.support_vectors_.ndim == 2
        assert reg.support_vectors_.shape[1] == X.shape[1]


# ---------------------------------------------------------------------------
# Common API invariants across all models
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelCls,kwargs,is_clf", [
    (CSVC, {"C": 1.0, "kernel": KERNEL_LINEAR}, True),
    (NuSVC, {"nu": 0.3, "kernel": KERNEL_LINEAR}, True),
    (EpsilonSVR, {"C": 1.0, "epsilon": 0.1, "kernel": KERNEL_LINEAR}, False),
    (NuSVR, {"C": 1.0, "nu": 0.5, "kernel": KERNEL_LINEAR}, False),
])
def test_fit_returns_self(ModelCls, kwargs, is_clf):
    rng = np.random.RandomState(42)
    X = rng.randn(20, 2)
    if is_clf:
        y = np.array([1] * 10 + [-1] * 10)
    else:
        y = rng.randn(20)
    model = ModelCls(**kwargs)
    result = model.fit(X, y)
    assert result is model
