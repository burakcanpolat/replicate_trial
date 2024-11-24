"""
Test module for the Replicate processor.

This module contains unit and integration tests for the ReplicateProcessor class.
It verifies the functionality of text processing, API token handling, error handling,
and file operations.

Test Categories:
    - Unit Tests: Test individual components with mocked dependencies
    - Integration Tests: Test end-to-end functionality with real API calls

Fixtures:
    - mock_api_response: Provides a mock for API responses
    - processor: Creates a ReplicateProcessor instance for testing

Dependencies:
    - pytest: Testing framework
    - pytest-timeout: For managing test timeouts
    - pytest-xdist: For parallel test execution
    - requests: For making HTTP requests
    - python-dotenv: For loading environment variables

Note:
    Integration tests require a valid REPLICATE_API_TOKEN in the .env file.
"""
import os
import pytest
import requests
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv
from replicate_trial.replicate_processor import ReplicateProcessor, ReplicateAPIError
import time
import json

# Load environment variables at the start of the test file
load_dotenv()

@pytest.fixture(scope="module")
def mock_api_response():
    """
    Fixture for mocking API responses.
    
    This fixture provides a mock for the requests.get function used in API calls.
    It's scoped at the module level for efficiency and returns a successful response
    by default.
    
    Returns:
        MagicMock: A mock object for requests.get with status_code=200
    """
    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        yield mock_get

@pytest.fixture(scope="module")
def processor(mock_api_response):
    """
    Fixture for creating a processor instance.
    
    Creates a ReplicateProcessor instance with a short timeout suitable for testing.
    Uses the mock_api_response fixture to avoid real API calls during initialization.
    
    Args:
        mock_api_response: The mock API response fixture
    
    Returns:
        ReplicateProcessor: A configured processor instance
    """
    return ReplicateProcessor(timeout=1)  # Short timeout for tests

@pytest.mark.unit
def test_replicate_processor_initialization(mock_api_response):
    """
    Test processor initialization with a valid API token.
    
    This test verifies that:
    1. The processor can be initialized with a valid API token
    2. The token is correctly stored in the processor instance
    3. Environment variables are properly handled
    
    Args:
        mock_api_response: Mock API response fixture
    """
    # Save original token
    original_token = os.environ.get("REPLICATE_API_TOKEN")
    os.environ["REPLICATE_API_TOKEN"] = "dummy_token"
    try:
        processor = ReplicateProcessor()
        assert processor.api_token == "dummy_token"
    finally:
        # Restore original token
        if original_token:
            os.environ["REPLICATE_API_TOKEN"] = original_token
        else:
            del os.environ["REPLICATE_API_TOKEN"]

@pytest.mark.unit
def test_replicate_processor_missing_token():
    """
    Test processor initialization with a missing API token.
    
    This test verifies that the processor raises a ValueError
    when the REPLICATE_API_TOKEN environment variable is not set.
    """
    with patch('os.getenv') as mock_getenv:
        # Mock getenv to return None for REPLICATE_API_TOKEN
        mock_getenv.return_value = None
        with pytest.raises(ValueError):
            ReplicateProcessor()

@pytest.mark.unit
def test_api_token_verification(mock_api_response):
    """
    Test API token verification functionality.
    
    This test verifies:
    1. Invalid tokens raise ReplicateAPIError
    2. Valid tokens allow successful initialization
    3. Token verification endpoint is correctly called
    
    Args:
        mock_api_response: Mock API response fixture
    """
    # Test with invalid token
    mock_api_response.side_effect = requests.exceptions.RequestException("Invalid token")
    with pytest.raises(ReplicateAPIError):
        ReplicateProcessor()
    
    # Test with valid token
    mock_api_response.side_effect = None
    mock_api_response.return_value.status_code = 200
    processor = ReplicateProcessor()
    assert processor._verify_api_token()

@pytest.mark.unit
def test_process_text_timeout(processor):
    """Test text processing timeout handling."""
    processor.timeout = 0.1  # Set a very short timeout
    
    def mock_stream(*args, **kwargs):
        time.sleep(0.2)  # Sleep longer than the timeout
        yield "Response that will never complete"
    
    with patch('replicate.stream') as mock_stream_func:
        mock_stream_func.return_value = mock_stream()
        with pytest.raises(TimeoutError) as exc_info:
            processor.process_text("Test text")
        assert "exceeded timeout" in str(exc_info.value)

@pytest.mark.unit
def test_process_text_retries(processor):
    """Test retry mechanism for failed API calls."""
    def failed_stream():
        raise ConnectionError("API call failed")
    
    def success_stream():
        yield json.dumps({
            "metadata": {
                "summary": "Test response",
                "tags": ["test"],
                "key_points": ["Test point"]
            },
            "formatted_text": "Test text"
        })
    
    with patch('replicate.stream') as mock_stream:
        # First two calls fail, third succeeds
        mock_stream.side_effect = [
            failed_stream(),
            failed_stream(),
            success_stream()
        ]
        
        result = processor.process_text("Test text", max_retries=3)
        assert result['metadata']['summary'] == "Test response"
        assert result['formatted_text'] == "Test text"

@pytest.mark.integration
@pytest.mark.skip(reason="Integration test requires valid API token")
def test_process_text_integration(processor):
    """
    Integration test for text processing functionality.
    
    This test verifies the end-to-end text processing workflow by:
    1. Sending a real request to the Replicate API
    2. Verifying that a valid response is received
    3. Checking the structure of the response
    
    Note:
        Requires a valid API token in the .env file.
    
    Args:
        processor: ReplicateProcessor fixture
    """
    # Test with a small sample text
    sample_text = "This is a test text. It needs formatting and analysis."
    result = processor.process_text(sample_text)
    
    # Check that we got a result
    assert isinstance(result, dict)
    assert 'formatted_text' in result or 'error' in result

@pytest.mark.unit
def test_save_processed_text(processor, tmp_path):
    """
    Test saving processed text to a file.

    This test verifies that:
    1. The output file is created successfully
    2. All sections (formatted text, summary, tags, key points) are written
    3. The file content matches the input data

    Args:
        processor: ReplicateProcessor fixture
        tmp_path: Pytest fixture for temporary directory
    """
    # Create a sample processed result
    processed_result = {
        'formatted_text': 'Sample formatted text',
        'metadata': {
            'summary': 'Sample summary',
            'tags': ['tag1', 'tag2'],
            'key_points': ['point1', 'point2']
        }
    }

    # Create temporary output file
    output_file = tmp_path / "output.txt"

    # Save the processed text
    processor.save_processed_text(processed_result, str(output_file))

    # Verify the file exists and contains the expected content
    assert output_file.exists()
    content = output_file.read_text()
    assert 'Sample formatted text' in content
    assert 'Sample summary' in content
    assert 'tag1, tag2' in content
    assert 'point1' in content
    assert 'point2' in content
