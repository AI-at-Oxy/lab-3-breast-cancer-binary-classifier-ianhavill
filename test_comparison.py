"""
Automated Tests for Model Comparison (Part 2)
COMP 395 – Deep Learning

DO NOT MODIFY THIS FILE.

Run with: python -m pytest test_comparison.py -v
"""

import ast
import importlib
import pytest


# =============================================================================
# Helper: static analysis of comparison.py
# =============================================================================

def _parse_source():
    """Read and parse comparison.py source code."""
    with open("comparison.py", "r") as f:
        source = f.read()
    tree = ast.parse(source)
    return source, tree


def _get_imported_modules(tree):
    """Extract all imported module names from the AST."""
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.add(node.module.split(".")[0])
    return modules


# =============================================================================
# Tests
# =============================================================================

class TestComparison:
    """Tests for the model comparison file."""

    def test_file_exists(self):
        """comparison.py should exist and be valid Python"""
        source, tree = _parse_source()
        assert tree is not None, "comparison.py could not be parsed"

    def test_has_description(self):
        """comparison.py should contain a descriptive comment or docstring (>50 chars)"""
        source, _ = _parse_source()
        # Count characters in comments and docstrings
        comment_chars = sum(
            len(line.strip()) - 1
            for line in source.splitlines()
            if line.strip().startswith("#")
        )
        # Also check for module-level docstring
        tree = ast.parse(source)
        docstring = ast.get_docstring(tree) or ""
        total_description = comment_chars + len(docstring)
        assert total_description >= 50, (
            "comparison.py should include a descriptive comment or docstring "
            "(at least 50 characters of comments found: "
            f"{total_description})"
        )

    def test_uses_sklearn(self):
        """comparison.py should import from sklearn"""
        _, tree = _parse_source()
        modules = _get_imported_modules(tree)
        assert "sklearn" in modules, (
            "comparison.py should import a model from sklearn"
        )

    def test_not_logistic_regression(self):
        """comparison.py should NOT use LogisticRegression"""
        source, _ = _parse_source()
        assert "LogisticRegression" not in source, (
            "Do not use LogisticRegression — that is essentially what you "
            "built from scratch. Choose a different model."
        )

    def test_uses_breast_cancer_data(self):
        """comparison.py should use the breast cancer dataset"""
        source, _ = _parse_source()
        uses_load_data = "load_data" in source
        uses_load_breast_cancer = "load_breast_cancer" in source
        assert uses_load_data or uses_load_breast_cancer, (
            "comparison.py should use the breast cancer dataset "
            "(via load_data from binary_classification or "
            "load_breast_cancer from sklearn)"
        )

    def test_runs_successfully(self):
        """comparison.py should run without errors"""
        import subprocess
        result = subprocess.run(
            ["python", "comparison.py"],
            capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, (
            f"comparison.py failed with error:\n{result.stderr}"
        )
