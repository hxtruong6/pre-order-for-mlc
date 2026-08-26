"""Smoke tests: the package imports and exposes its public API."""

import importlib.metadata

import preorder4mlc


def test_package_imports():
    assert preorder4mlc is not None


def test_version_matches_metadata():
    assert importlib.metadata.version("preorder4mlc") == "1.0.2"
