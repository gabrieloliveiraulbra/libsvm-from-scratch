"""conftest.py — pytest configuration for antigravity-svm tests."""
import sys
import os

# Make the package importable without installing it
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
