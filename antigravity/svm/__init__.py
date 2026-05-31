"""
antigravity.svm
===============
A pure NumPy implementation of Support Vector Machines based on the
LIBSVM library (Chang & Lin, 2011).

Submodules
----------
kernels     : Kernel functions (linear, polynomial, RBF, sigmoid)
solver      : SMO-based QP solver with WSS, LRU cache and shrinking
models      : C-SVC, ν-SVC, One-Class SVM, ε-SVR, ν-SVR
multiclass  : One-Against-One multiclass strategy
probability : Platt scaling and Laplace probability estimation
tuning      : Grid-search cross-validation hyperparameter tuner

References
----------
Chang, C.-C., & Lin, C.-J. (2011).
    LIBSVM: A library for support vector machines.
    ACM Transactions on Intelligent Systems and Technology, 2(3), 27:1–27:27.
    https://doi.org/10.1145/1961189.1961199
"""

from antigravity.svm.kernels import (
    linear_kernel,
    polynomial_kernel,
    rbf_kernel,
    sigmoid_kernel,
    compute_kernel_matrix,
    KERNEL_LINEAR,
    KERNEL_POLY,
    KERNEL_RBF,
    KERNEL_SIGMOID,
)
from antigravity.svm.models import CSVC, NuSVC, OneClassSVM, EpsilonSVR, NuSVR
from antigravity.svm.multiclass import MulticlassSVC
from antigravity.svm.tuning import GridSearchCV

__all__ = [
    # Kernel constants
    "KERNEL_LINEAR",
    "KERNEL_POLY",
    "KERNEL_RBF",
    "KERNEL_SIGMOID",
    # Kernel functions
    "linear_kernel",
    "polynomial_kernel",
    "rbf_kernel",
    "sigmoid_kernel",
    "compute_kernel_matrix",
    # Models
    "CSVC",
    "NuSVC",
    "OneClassSVM",
    "EpsilonSVR",
    "NuSVR",
    # Multiclass
    "MulticlassSVC",
    # Tuning
    "GridSearchCV",
]
