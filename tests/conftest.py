"""Test configuration and shared fixtures for replicate_trial."""

import os
import pytest
from pathlib import Path


@pytest.fixture
def test_data_dir():
    """Return the path to the test data directory."""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def sample_text_file(test_data_dir):
    """Return the path to a sample text file for testing."""
    return test_data_dir / "test_sample.txt"


@pytest.fixture
def mock_env(monkeypatch):
    """Mock environment variables for testing."""
    monkeypatch.setenv("REPLICATE_API_TOKEN", "test_token")
    return {"REPLICATE_API_TOKEN": "test_token"}
