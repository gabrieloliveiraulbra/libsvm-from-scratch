"""
setup.py for antigravity-svm
"""
from setuptools import setup, find_packages

setup(
    name="antigravity-svm",
    version="0.1.0",
    description=(
        "A pure NumPy implementation of Support Vector Machines "
        "based on the LIBSVM library (Chang & Lin, 2011)."
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="antigravity",
    packages=find_packages(exclude=["tests*", "examples*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
