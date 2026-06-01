"""
antigravity.svm — end-to-end demonstration.

Runs five scenarios corresponding to the five SVM formulations in the
LIBSVM paper (Chang & Lin, 2011), plus multiclass OAO, Platt scaling,
Laplace SVR, and grid-search.
"""

from __future__ import annotations
import numpy as np

# ── helpers ──────────────────────────────────────────────────────────────────

def separator(title: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(y_true == y_pred))


def mse(y_true, y_pred) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


# ── data generators ──────────────────────────────────────────────────────────

def make_binary(n: int = 60, seed: int = 0):
    rng = np.random.RandomState(seed)
    X_p = rng.randn(n // 2, 2) + np.array([2.0, 2.0])
    X_n = rng.randn(n // 2, 2) + np.array([-2.0, -2.0])
    X = np.vstack([X_p, X_n])
    y = np.array([1] * (n // 2) + [-1] * (n // 2))
    return X, y


def make_multiclass(k: int = 4, n_per_class: int = 25, seed: int = 7):
    rng = np.random.RandomState(seed)
    centres = np.array([[0, 0], [8, 0], [0, 8], [8, 8]])[:k]
    X = np.vstack([rng.randn(n_per_class, 2) + c for c in centres])
    y = np.repeat(np.arange(k), n_per_class)
    return X, y


def make_regression(n: int = 50, seed: int = 3):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 2)
    y = 3.0 * X[:, 0] - 1.5 * X[:, 1] + 0.3 * rng.randn(n)
    return X, y


def make_inlier_outlier(n_train: int = 80, seed: int = 5):
    rng = np.random.RandomState(seed)
    X_train = rng.randn(n_train, 2)
    X_in = rng.randn(20, 2)
    X_out = rng.randn(20, 2) + 6.0
    return X_train, X_in, X_out


# =============================================================================
# 1. C-SVC — binary classification
# =============================================================================

separator("1. C-SVC  (C-Support Vector Classification)")

from antigravity.svm import CSVC, KERNEL_RBF, KERNEL_LINEAR

X, y = make_binary()
clf = CSVC(C=5.0, kernel=KERNEL_RBF, gamma=0.5, tol=1e-4)
clf.fit(X, y)
print(f"  Training accuracy : {accuracy(y, clf.predict(X)):.2%}")
print(f"  # Support Vectors : {len(clf.support_)}")
print(f"  Intercept (b)     : {clf.intercept_:.4f}")


# =============================================================================
# 2. ν-SVC — nu-classification
# =============================================================================

separator("2. ν-SVC  (Nu-Support Vector Classification)")

from antigravity.svm import NuSVC

clf_nu = NuSVC(nu=0.2, kernel=KERNEL_RBF, gamma=0.5, tol=1e-4)
clf_nu.fit(X, y)
print(f"  Training accuracy : {accuracy(y, clf_nu.predict(X)):.2%}")
print(f"  # Support Vectors : {len(clf_nu.support_)}")


# =============================================================================
# 3. One-Class SVM — novelty detection
# =============================================================================

separator("3. One-Class SVM  (Novelty / Outlier Detection)")

from antigravity.svm import OneClassSVM

X_train, X_in, X_out = make_inlier_outlier()
oc = OneClassSVM(nu=0.05, kernel=KERNEL_RBF, gamma=0.5, tol=1e-4)
oc.fit(X_train)
frac_in = np.mean(oc.predict(X_in) == 1)
frac_out = np.mean(oc.predict(X_out) == -1)
print(f"  Inlier detection rate  : {frac_in:.0%}")
print(f"  Outlier detection rate : {frac_out:.0%}")


# =============================================================================
# 4. ε-SVR — epsilon-regression
# =============================================================================

separator("4. ε-SVR  (Epsilon-Support Vector Regression)")

from antigravity.svm import EpsilonSVR

X_r, y_r = make_regression()
reg_e = EpsilonSVR(C=10.0, epsilon=0.1, kernel=KERNEL_RBF, gamma=0.5, tol=1e-4)
reg_e.fit(X_r, y_r)
print(f"  Training MSE      : {mse(y_r, reg_e.predict(X_r)):.4f}")
print(f"  # Support Vectors : {len(reg_e.support_)}")


# =============================================================================
# 5. ν-SVR — nu-regression
# =============================================================================

separator("5. ν-SVR  (Nu-Support Vector Regression)")

from antigravity.svm import NuSVR

reg_nu = NuSVR(C=10.0, nu=0.3, kernel=KERNEL_RBF, gamma=0.5, tol=1e-4)
reg_nu.fit(X_r, y_r)
print(f"  Training MSE      : {mse(y_r, reg_nu.predict(X_r)):.4f}")
print(f"  # Support Vectors : {len(reg_nu.support_)}")


# =============================================================================
# 6. Multiclass — One-Against-One
# =============================================================================

separator("6. MulticlassSVC  (One-Against-One, 4 classes)")

from antigravity.svm import MulticlassSVC

X_m, y_m = make_multiclass(k=4)
mc = MulticlassSVC(C=5.0, kernel=KERNEL_RBF, gamma=0.2, tol=1e-4)
mc.fit(X_m, y_m)
print(f"  # Binary classifiers : {len(mc.classifiers_)}")
print(f"  Training accuracy    : {accuracy(y_m, mc.predict(X_m)):.2%}")
proba = mc.predict_proba(X_m)
print(f"  Prob row-sum (first) : {proba[0].sum():.6f}  (should be 1.0)")


# =============================================================================
# 7. Platt scaling — calibrated probabilities
# =============================================================================

separator("7. Platt Scaling  (Calibrated Probabilities)")

from antigravity.svm.probability import PlattScaling

X, y = make_binary()
clf_base = CSVC(C=5.0, kernel=KERNEL_RBF, gamma=0.5, tol=1e-4)
clf_base.fit(X, y)
df = clf_base.decision_function(X)

platt = PlattScaling(n_folds=3)
platt.fit(df, y.astype(float))
p_pos = platt.predict_proba_binary(df)
print(f"  A = {platt.A_:.4f},  B = {platt.B_:.4f}")
print(f"  P(y=+1) range : [{p_pos.min():.3f}, {p_pos.max():.3f}]")
print(f"  Samples with P > 0.5 classified +1: "
      f"{np.mean((p_pos > 0.5) == (y == 1)):.2%}")


# =============================================================================
# 8. Laplace SVR — prediction intervals
# =============================================================================

separator("8. LaplaceSVR  (Prediction Intervals)")

from antigravity.svm.probability import LaplaceSVR

X_r, y_r = make_regression()
reg = EpsilonSVR(C=10.0, kernel=KERNEL_RBF, gamma=0.5, tol=1e-4)
reg.fit(X_r, y_r)
residuals = np.abs(y_r - reg.predict(X_r))

lap = LaplaceSVR()
lap.fit(residuals)
lo, hi = lap.predict_interval(reg.predict(X_r), confidence=0.95)
coverage = np.mean((y_r >= lo) & (y_r <= hi))
print(f"  Laplace σ̂     : {lap.sigma_:.4f}")
print(f"  95% PI coverage (train) : {coverage:.2%}  (expect ≥ 95%)")


# =============================================================================
# 9. Grid Search — hyperparameter tuning
# =============================================================================

separator("9. GridSearchCV  (C × γ grid, 5-fold CV)")

from antigravity.svm import GridSearchCV

X, y = make_binary()
param_grid = {
    "C":     [0.1, 1.0, 10.0],
    "gamma": [0.1, 0.5, 2.0],
}
gs = GridSearchCV(
    CSVC, param_grid, cv=5, scoring="accuracy",
    random_state=42,
    fixed_params={"kernel": KERNEL_RBF},
)
gs.fit(X, y)
print(gs.summary())


print("\n✓ All demonstrations completed successfully.")
