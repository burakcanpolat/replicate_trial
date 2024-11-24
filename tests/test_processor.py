"""Basic tests for the ReplicateProcessor class."""

import pytest
import json
from unittest.mock import patch, MagicMock
from pathlib import Path
from replicate_trial.replicate_processor import ReplicateProcessor, ReplicateAPIError
from replicate_trial.rate_limiter import RateLimiter

@pytest.fixture
def mock_env(monkeypatch):
    """Mock environment variables."""
    monkeypatch.setenv("REPLICATE_API_TOKEN", "test_token")
    return None

@pytest.fixture
def mock_requests_get(mocker):
    """Mock requests.get to return a successful response."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status.return_value = None
    mock = mocker.patch('requests.get', return_value=mock_response)
    return mock

@pytest.fixture
def mock_replicate_stream(mocker):
    """Mock the replicate.stream function to return a valid response."""
    def mock_stream(*args, **kwargs):
        input_text = kwargs['input']['prompt'].split('Text to process:\n\n')[-1]
        json_response = {
            "metadata": {
                "summary": "Test response",
                "tags": ["test"],
                "key_points": ["Test point"]
            },
            "formatted_text": input_text
        }
        yield json.dumps(json_response)
    
    mock = mocker.patch("replicate.stream", side_effect=mock_stream)
    return mock

@pytest.fixture
def processor(mock_env, mock_replicate_stream, mock_requests_get):
    """Create a ReplicateProcessor instance with mocked dependencies."""
    return ReplicateProcessor()

def test_basic_text_processing(processor):
    """Test basic text processing functionality."""
    test_text = "This is a basic test."
    result = processor.process_text(test_text)
    assert result['formatted_text'] == test_text
    assert 'metadata' in result
    assert 'summary' in result['metadata']
    assert 'tags' in result['metadata']
    assert 'key_points' in result['metadata']
