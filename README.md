# antigravity.svm

A pure **NumPy** implementation of Support Vector Machines based on the LIBSVM library.

> Chang, C.-C., & Lin, C.-J. (2011). **LIBSVM: A library for support vector machines**.  
> *ACM Transactions on Intelligent Systems and Technology*, 2(3), 27:1–27:27.

---

## Features

| Module | Contents |
|--------|----------|
| `antigravity.svm.kernels` | Linear, Polynomial, RBF, Sigmoid |
| `antigravity.svm.solver` | SMO + WSS-3 + LRU cache + Shrinking + KKT |
| `antigravity.svm.models` | C-SVC, ν-SVC, One-Class SVM, ε-SVR, ν-SVR |
| `antigravity.svm.multiclass` | One-Against-One (k(k-1)/2 binary classifiers) |
| `antigravity.svm.probability` | Platt scaling + Laplace SVR intervals |
| `antigravity.svm.tuning` | Grid Search + k-fold Cross-Validation |

**Dependencies**: `numpy >= 1.21` only (no scipy, no sklearn).

---

## Installation

```bash
pip install -e .
```

---

## Quick Start

### Binary Classification (C-SVC)

```python
import numpy as np
from antigravity.svm import CSVC, KERNEL_RBF

X_train = np.random.randn(100, 2)
y_train = np.sign(X_train[:, 0])

clf = CSVC(C=1.0, kernel=KERNEL_RBF, gamma=0.5)
clf.fit(X_train, y_train)
print(clf.predict(X_train[:5]))
```

### Multiclass (One-vs-One)

```python
from antigravity.svm import MulticlassSVC

mc = MulticlassSVC(C=5.0, kernel=KERNEL_RBF, gamma=0.3)
mc.fit(X_train, y_multiclass)
print(mc.predict(X_test))
```

### Grid Search

```python
from antigravity.svm import CSVC, GridSearchCV, KERNEL_RBF

gs = GridSearchCV(
    CSVC,
    {"C": [0.1, 1.0, 10.0], "gamma": [0.01, 0.1, 1.0]},
    cv=5,
    scoring="accuracy",
    fixed_params={"kernel": KERNEL_RBF},
)
gs.fit(X_train, y_train)
print(gs.best_params_, gs.best_score_)
print(gs.summary())
```

### Probability Calibration (Platt Scaling)

```python
from antigravity.svm.probability import PlattScaling

platt = PlattScaling(n_folds=5)
platt.fit(clf.decision_function(X_train), y_train.astype(float))
proba = platt.predict_proba(clf.decision_function(X_test))
```

### Regression (ε-SVR) with Prediction Intervals

```python
from antigravity.svm import EpsilonSVR
from antigravity.svm.probability import LaplaceSVR

reg = EpsilonSVR(C=10.0, epsilon=0.1, kernel=KERNEL_RBF, gamma=0.5)
reg.fit(X_train, y_train)

lap = LaplaceSVR(n_folds=5)
lap.fit_with_cv(lambda: EpsilonSVR(C=10.0, epsilon=0.1), X_train, y_train)
lower, upper = lap.predict_interval(reg.predict(X_test), confidence=0.95)
```

---

## Running Tests

```bash
pytest tests/ -v
```

## Running the Demo

```bash
python examples/demo.py
```

---

## Mathematical Formulations

### C-SVC Dual (Eq. 1)

```
min   ½ αᵀ Q α − eᵀ α
s.t.  yᵀ α = 0,   0 ≤ αᵢ ≤ C
      Q_ij = y_i y_j K(x_i, x_j)
```

### ν-SVC Dual (Eq. 3)

```
min   ½ ᾱᵀ Q ᾱ
s.t.  yᵀ ᾱ = 0,   eᵀ ᾱ = ν,   0 ≤ ᾱᵢ ≤ 1/l
```

### ε-SVR Dual (Eq. 5)

```
min   ½ (α−α*)ᵀ Q (α−α*) + ε eᵀ(α+α*) − yᵀ(α−α*)
s.t.  eᵀ(α−α*) = 0,   0 ≤ αᵢ, αᵢ* ≤ C
```

### SMO / WSS-3

The solver uses **second-order working set selection** (Fan, Chen & Lin, 2005):

1. Select *i* maximising −yᵢ ∇fᵢ over I_up
2. Select *j* minimising the quadratic decrease:
   `− (∇f_i − ∇f_j)² / (Q_ii + Q_jj − 2 Q_ij)`

Convergence is declared when the KKT gap:

```
max_{I_up}(−y·∇f) − min_{I_low}(−y·∇f) ≤ ε
```

---

## License

MIT
