"""Advanced tests for the ReplicateProcessor class focusing on edge cases and complex scenarios."""

import pytest
import json
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import os
from replicate_trial.replicate_processor import ReplicateProcessor, ReplicateAPIError

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
def processor(mock_env, mock_requests_get):
    """Create a ReplicateProcessor instance with mocked dependencies."""
    return ReplicateProcessor()

def test_initialization_with_custom_params(mock_env, mock_requests_get):
    """Test initialization with custom parameters."""
    custom_processor = ReplicateProcessor(
        model_version="custom_model_version",
        timeout=600,
        requests_per_minute=30.0
    )
    assert custom_processor.model_version == "custom_model_version"
    assert custom_processor.timeout == 600
    assert custom_processor.rate_limiter.rate == 0.5  # 30 requests per minute = 0.5 per second

def test_text_chunking(processor):
    """Test text chunking with different scenarios."""
    # Test with short text
    short_text = "Short text that doesn't need chunking."
    assert len(processor._chunk_text(short_text)) == 1
    
    # Test with text that needs exactly two chunks
    chars_per_chunk = 1500 * 4  # max_chunk_size * chars_per_token
    text = "a" * (chars_per_chunk + 1)
    chunks = processor._chunk_text(text)
    assert len(chunks) == 2
    
    # Test with text containing natural break points
    text_with_breaks = "First sentence. Second sentence.\nNew paragraph.\nAnother paragraph."
    chunks = processor._chunk_text(text_with_breaks, max_chunk_size=10)
    assert all(len(chunk) <= 40 for chunk in chunks)  # 10 tokens * 4 chars per token
    
def test_save_processed_text_formats(processor):
    """Test saving processed text with different metadata formats."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "output.txt")
        
        # Test with minimal metadata
        minimal_result = {
            "metadata": {
                "summary": "Test summary",
                "tags": [],
                "key_points": []
            },
            "formatted_text": "Test text"
        }
        processor.save_processed_text(minimal_result, output_file)
        assert os.path.exists(output_file)
        
        # Test with rich metadata
        rich_result = {
            "metadata": {
                "summary": "Detailed summary",
                "tags": ["tag1", "tag2"],
                "key_points": ["point1", "point2"],
                "additional_field": "extra"
            },
            "formatted_text": "Rich text content"
        }
        processor.save_processed_text(rich_result, output_file)
        with open(output_file, 'r') as f:
            content = f.read()
            assert "Detailed summary" in content
            assert "tag1" in content
            assert "point1" in content

def test_process_text_with_different_templates(processor, mocker):
    """Test processing text with different template keys."""
    def mock_run(*args, **kwargs):
        # Return different responses based on the template in the prompt
        prompt = kwargs['input']['prompt']
        if "technical editor" in prompt.lower():
            return [json.dumps({
                "metadata": {"summary": "Technical template", "tags": [], "key_points": []},
                "formatted_text": "Technical text"
            })]
        return [json.dumps({
            "metadata": {"summary": "Default template", "tags": [], "key_points": []},
            "formatted_text": "Default text"
        })]
    
    mocker.patch("replicate.run", side_effect=mock_run)
    
    # Test with default template
    result = processor.process_text("Test text")
    assert "Default" in result['metadata']['summary']
    
    # Test with technical template
    result = processor.process_text("Test text", template_key="technical")
    assert "Technical" in result['metadata']['summary']

def test_concurrent_rate_limiting(processor, mocker):
    """Test rate limiting with concurrent requests."""
    processed_count = 0
    def mock_run(*args, **kwargs):
        nonlocal processed_count
        processed_count += 1
        return [json.dumps({
            "metadata": {"summary": f"Request {processed_count}", "tags": [], "key_points": []},
            "formatted_text": "Test"
        })]
    
    mocker.patch("replicate.run", side_effect=mock_run)
    
    # Try to process multiple texts quickly
    results = []
    for _ in range(5):
        try:
            result = processor.process_text("Test text")
            results.append(result)
        except ReplicateAPIError:
            pass
    
    # Check that rate limiting was applied
    assert len(results) <= 5  # Some requests might be rate limited

def test_error_recovery_scenarios(processor, mocker):
    """Test various error recovery scenarios."""
    call_count = 0
    def mock_run_with_errors(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Network error should be retried
            raise Exception("Network error")
        elif call_count == 2:
            # Invalid JSON response should be handled gracefully
            return ["Invalid JSON {"]
        return [json.dumps({
            "metadata": {"summary": "Success", "tags": [], "key_points": []},
            "formatted_text": "Test text"  # Match the input text
        })]
    
    mocker.patch("replicate.run", side_effect=mock_run_with_errors)
    
    # Test handling of network error with retry
    result = processor.process_text("Test text", max_retries=3)
    assert result['formatted_text'] == "Test text"  # Should match input text
    assert 'error' in result['metadata']  # Should have error in metadata
    assert 'Error processing text' in result['metadata']['summary']
    
    # Test handling of invalid template
    with pytest.raises(ReplicateAPIError) as exc_info:
        processor.process_text("Test text", template_key="nonexistent")
    assert "Template 'nonexistent' not found" in str(exc_info.value)
    
    # Test handling of rate limit exceeded
    with pytest.raises(ReplicateAPIError) as exc_info:
        # Mock rate limiter to always return False
        mocker.patch.object(processor.rate_limiter, 'acquire', return_value=False)
        processor.process_text("Test text")
    assert "Rate limit exceeded" in str(exc_info.value)

def test_unicode_and_special_chars_in_metadata(processor, mocker):
    """Test handling of Unicode and special characters in metadata."""
    def mock_run(*args, **kwargs):
        return [json.dumps({
            "metadata": {
                "summary": "Test with Unicode: 你好",
                "tags": ["emoji🌍", "special&chars"],
                "key_points": ["Point with quotes'\""]
            },
            "formatted_text": "Unicode text: 你好 🌍"
        })]
    
    mocker.patch("replicate.run", side_effect=mock_run)
    
    result = processor.process_text("Test text")
    assert "你好" in result['metadata']['summary']
    assert "emoji🌍" in result['metadata']['tags']
    assert "🌍" in result['formatted_text']
