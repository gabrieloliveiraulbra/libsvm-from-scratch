"""Tests for antigravity.svm.multiclass"""
import numpy as np
import pytest
from antigravity.svm.multiclass import MulticlassSVC
from antigravity.svm.kernels import KERNEL_RBF, KERNEL_LINEAR


@pytest.fixture
def three_class_dataset():
    """Well-separated 3-class dataset."""
    rng = np.random.RandomState(99)
    centres = np.array([[0, 0], [5, 0], [2.5, 4.5]])
    X = np.vstack([rng.randn(30, 2) + c for c in centres])
    y = np.repeat([0, 1, 2], 30)
    return X, y


@pytest.fixture
def four_class_dataset():
    """4-class dataset with clear cluster separation."""
    rng = np.random.RandomState(77)
    centres = np.array([[0, 0], [8, 0], [0, 8], [8, 8]])
    X = np.vstack([rng.randn(25, 2) + c for c in centres])
    y = np.repeat([10, 20, 30, 40], 25)
    return X, y


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestMulticlassSVC:
    def test_fit_predict_3class(self, three_class_dataset):
        X, y = three_class_dataset
        clf = MulticlassSVC(C=1.0, kernel=KERNEL_RBF, gamma=0.5)
        clf.fit(X, y)
        y_pred = clf.predict(X)
        accuracy = np.mean(y_pred == y)
        assert accuracy >= 0.90, f"Expected >= 90% accuracy, got {accuracy:.2%}"

    def test_number_of_classifiers_3class(self, three_class_dataset):
        X, y = three_class_dataset
        clf = MulticlassSVC(C=1.0, kernel=KERNEL_LINEAR)
        clf.fit(X, y)
        # k(k-1)/2 = 3*2/2 = 3 classifiers
        assert len(clf.classifiers_) == 3
        assert len(clf.pairs_) == 3

    def test_number_of_classifiers_4class(self, four_class_dataset):
        X, y = four_class_dataset
        clf = MulticlassSVC(C=1.0, kernel=KERNEL_LINEAR)
        clf.fit(X, y)
        # k(k-1)/2 = 4*3/2 = 6 classifiers
        assert len(clf.classifiers_) == 6

    def test_classes_stored(self, three_class_dataset):
        X, y = three_class_dataset
        clf = MulticlassSVC()
        clf.fit(X, y)
        np.testing.assert_array_equal(clf.classes_, [0, 1, 2])

    def test_prediction_within_classes(self, three_class_dataset):
        X, y = three_class_dataset
        clf = MulticlassSVC(C=1.0, kernel=KERNEL_RBF, gamma=0.5)
        clf.fit(X, y)
        y_pred = clf.predict(X)
        assert set(y_pred).issubset(set(clf.classes_))

    def test_predict_shape(self, three_class_dataset):
        X, y = three_class_dataset
        clf = MulticlassSVC(C=1.0, kernel=KERNEL_LINEAR)
        clf.fit(X, y)
        X_test = np.random.randn(10, 2)
        y_pred = clf.predict(X_test)
        assert y_pred.shape == (10,)

    def test_4class_accuracy(self, four_class_dataset):
        X, y = four_class_dataset
        clf = MulticlassSVC(C=10.0, kernel=KERNEL_RBF, gamma=0.1)
        clf.fit(X, y)
        accuracy = np.mean(clf.predict(X) == y)
        assert accuracy >= 0.90

    def test_original_labels_preserved(self, four_class_dataset):
        X, y = four_class_dataset
        clf = MulticlassSVC(C=1.0, kernel=KERNEL_LINEAR)
        clf.fit(X, y)
        y_pred = clf.predict(X)
        assert set(y_pred).issubset({10, 20, 30, 40})

    def test_string_labels(self):
        rng = np.random.RandomState(42)
        X = np.vstack([
            rng.randn(15, 2) + [0, 0],
            rng.randn(15, 2) + [5, 5],
            rng.randn(15, 2) + [10, 0],
        ])
        y = np.array(["alpha"] * 15 + ["beta"] * 15 + ["gamma"] * 15)
        clf = MulticlassSVC(C=1.0, kernel=KERNEL_RBF, gamma=0.5)
        clf.fit(X, y)
        y_pred = clf.predict(X)
        assert set(y_pred).issubset({"alpha", "beta", "gamma"})
        assert np.mean(y_pred == y) >= 0.85

    def test_predict_proba_sums_to_one(self, three_class_dataset):
        X, y = three_class_dataset
        clf = MulticlassSVC(C=1.0, kernel=KERNEL_LINEAR)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)

    def test_decision_function_shape(self, three_class_dataset):
        X, y = three_class_dataset
        clf = MulticlassSVC(C=1.0, kernel=KERNEL_LINEAR)
        clf.fit(X, y)
        votes = clf.decision_function(X)
        assert votes.shape == (len(y), 3)

    def test_two_classes_raises_if_only_one(self):
        X = np.random.randn(10, 2)
        y = np.zeros(10, dtype=int)
        with pytest.raises(ValueError, match="at least 2"):
            MulticlassSVC().fit(X, y)

    def test_not_fitted_raises(self):
        clf = MulticlassSVC()
        with pytest.raises(RuntimeError):
            clf.predict(np.ones((3, 2)))

    def test_fit_returns_self(self, three_class_dataset):
        X, y = three_class_dataset
        clf = MulticlassSVC()
        assert clf.fit(X, y) is clf
