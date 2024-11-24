"""
Test module for text formatting functionality.

This module focuses on testing the text formatting capabilities of the ReplicateProcessor,
ensuring that text structure, spacing, and special characters are preserved correctly.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from replicate_trial.replicate_processor import ReplicateProcessor
from replicate_trial.prompt_templates import TEMPLATES
import json

@pytest.fixture
def mock_env(monkeypatch):
    """Mock environment variables."""
    monkeypatch.setenv("REPLICATE_API_TOKEN", "test_token")

@pytest.fixture
def mock_response():
    """Create a mock response for requests."""
    mock = MagicMock()
    mock.status_code = 200
    mock.raise_for_status.return_value = None
    return mock

@pytest.fixture
def mock_replicate_output():
    """Mock the replicate.run output."""
    return ['{"metadata": {"summary": "Test summary", "tags": ["test"], "key_points": ["test point"]}, "formatted_text": "Test text"}']

@pytest.fixture
def mock_replicate_run(mocker):
    """Mock the replicate.run function to return a valid response."""
    def wrap_text(text: str, width: int = 100) -> str:
        """Wrap text at word boundaries."""
        lines = []
        for paragraph in text.split('\n'):
            if not paragraph.strip():
                lines.append('')
                continue
                
            # Preserve indentation for lists
            indent = len(paragraph) - len(paragraph.lstrip())
            indentation = paragraph[:indent]
            words = paragraph[indent:].split()
            
            current_line = [indentation]
            current_length = indent
            
            for word in words:
                word_length = len(word)
                if current_length + 1 + word_length <= width:
                    if current_line[-1] != indentation:
                        current_line.append(' ')
                        current_length += 1
                    current_line.append(word)
                    current_length += word_length
                else:
                    lines.append(''.join(current_line))
                    current_line = [indentation, word]
                    current_length = indent + word_length
                    
            if current_line:
                lines.append(''.join(current_line))
                
        return '\n'.join(lines)

    def mock_run(*args, **kwargs):
        input_text = kwargs['input']['prompt'].split('Text to process:\n\n')[-1]
        wrapped_text = wrap_text(input_text)
        return [json.dumps({
            "metadata": {
                "summary": "Test text about formatting and preservation",
                "tags": ["test", "formatting", "preservation"],
                "key_points": ["Contains test text", "Tests formatting preservation"]
            },
            "formatted_text": wrapped_text
        })]
    
    mock = mocker.patch("replicate.run", side_effect=mock_run)
    return mock

@pytest.fixture
def mock_requests_get(mocker):
    """Mock requests.get to return a successful response."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status.return_value = None
    mock = mocker.patch('requests.get', return_value=mock_response)
    return mock

@pytest.fixture
def processor(mock_env, mock_replicate_run, mock_requests_get):
    """Create a ReplicateProcessor instance with mocked dependencies."""
    return ReplicateProcessor()

@pytest.fixture
def test_text():
    """Load the formatting test text file."""
    test_file = Path(__file__).parent / "test_data" / "formatting_test.txt"
    with open(test_file, 'r', encoding='utf-8') as f:
        return f.read()

def test_text_structure_preservation(processor, test_text):
    """
    Test that text structure (paragraphs, lists) is preserved.
    
    This test verifies that:
    1. Paragraph breaks are maintained
    2. Bullet points and numbered lists keep their format
    3. Line breaks occur at appropriate points
    """
    result = processor.process_text(test_text)
    formatted_text = result['formatted_text']
    
    # Check paragraph separation
    assert "\n\n" in formatted_text
    
    # Check bullet points preservation
    assert "* Here's a bullet point" in formatted_text
    assert "* And another one" in formatted_text
    
    # Check numbered list preservation
    assert "1. This is a numbered list" in formatted_text
    assert "2. It should keep the numbers" in formatted_text

def test_special_characters_preservation(processor, test_text):
    """
    Test that special characters are preserved correctly.
    
    This test verifies that:
    1. Punctuation marks are preserved
    2. Special characters (@#$%^&*) are maintained
    3. Spacing around special characters is correct
    """
    result = processor.process_text(test_text)
    formatted_text = result['formatted_text']
    
    # Check special characters
    assert "@#$%^&*()" in formatted_text
    
    # Check punctuation
    assert "quick brown fox jumps over" in formatted_text
    assert "commonly used as a pangram" in formatted_text

def test_word_order_preservation(processor, test_text):
    """
    Test that all words are preserved in their original order.
    
    This test verifies that:
    1. No words are omitted
    2. Word order remains unchanged
    3. Sentence structure is maintained
    """
    result = processor.process_text(test_text)
    formatted_text = result['formatted_text']
    
    # Convert to word lists (ignoring whitespace and case)
    original_words = [w.lower() for w in test_text.split() if w.strip()]
    formatted_words = [w.lower() for w in formatted_text.split() if w.strip()]
    
    # Check word preservation and order
    assert len(original_words) == len(formatted_words)
    assert original_words == formatted_words

def test_line_wrapping(processor, test_text):
    """
    Test that text is wrapped appropriately.
    
    This test verifies that:
    1. Lines are wrapped at appropriate lengths
    2. Words are not broken mid-word
    3. Paragraph structure is maintained after wrapping
    """
    result = processor.process_text(test_text)
    formatted_text = result['formatted_text']
    
    # Check that no lines are excessively long
    max_line_length = 100
    lines = formatted_text.split('\n')
    for line in lines:
        assert len(line.strip()) <= max_line_length
        
    # Check that words aren't broken
    all_words = []
    for line in lines:
        words = line.strip().split()
        if words:
            # Check that no word contains a hyphen at the end
            assert not any(w.endswith('-') for w in words[:-1])
            all_words.extend(words)
    
    # Verify all original words are present
    original_words = [w for w in test_text.split() if w.strip()]
    assert len(all_words) == len(original_words)

def test_metadata_extraction(processor, test_text):
    """
    Test that metadata is correctly extracted from the text.
    
    This test verifies that:
    1. Summary is generated
    2. Tags are relevant
    3. Key points are extracted
    """
    result = processor.process_text(test_text)
    
    # Check metadata structure
    assert 'metadata' in result
    assert 'summary' in result['metadata']
    assert 'tags' in result['metadata']
    assert 'key_points' in result['metadata']
    
    # Check metadata content
    metadata = result['metadata']
    assert isinstance(metadata['summary'], str)
    assert len(metadata['summary']) > 0
    
    assert isinstance(metadata['tags'], list)
    assert len(metadata['tags']) > 0
    
    assert isinstance(metadata['key_points'], list)
    assert len(metadata['key_points']) > 0
