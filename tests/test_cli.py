"""Tests for the CLI module."""

import os
import json
import pytest
from pathlib import Path
from click.testing import CliRunner
from unittest.mock import patch, MagicMock

from replicate_trial.cli import cli, process, estimate
from replicate_trial.replicate_processor import ReplicateProcessor

@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()

@pytest.fixture
def mock_processor():
    """Create a mock ReplicateProcessor."""
    with patch('replicate_trial.cli.ReplicateProcessor') as mock:
        processor_instance = MagicMock()
        processor_instance.process_text.return_value = {
            'metadata': {
                'summary': 'Test summary',
                'key_points': ['Point 1', 'Point 2'],
                'tags': ['tag1', 'tag2'],
            },
            'formatted_text': 'Formatted test content'
        }
        mock.return_value = processor_instance
        yield mock

@pytest.fixture
def sample_text_file(tmp_path):
    """Create a sample text file for testing."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("This is a test document.\nIt has multiple lines.\nFor testing purposes.")
    return str(file_path)

@pytest.fixture
def output_dir(tmp_path):
    """Create a temporary output directory."""
    output_path = tmp_path / "output"
    output_path.mkdir()
    return str(output_path)

def test_process_txt_format(runner, mock_processor, sample_text_file, output_dir):
    """Test processing a file with txt output format."""
    result = runner.invoke(cli, ['process', sample_text_file, '-o', output_dir, '--format', 'txt'])
    assert result.exit_code == 0
    
    # Check if output file exists
    output_file = Path(output_dir) / "test_output.txt"
    assert output_file.exists()
    
    # Verify content format
    content = output_file.read_text()
    assert "METADATA" in content
    assert "Summary:" in content
    assert "Key Points:" in content

def test_process_json_format(runner, mock_processor, sample_text_file, output_dir):
    """Test processing a file with json output format."""
    result = runner.invoke(cli, ['process', sample_text_file, '-o', output_dir, '--format', 'json'])
    assert result.exit_code == 0
    
    # Check if output file exists
    output_file = Path(output_dir) / "test_output.json"
    assert output_file.exists()
    
    # Verify JSON content
    with open(output_file) as f:
        content = json.load(f)
    assert 'metadata' in content
    assert 'formatted_text' in content

def test_process_both_formats(runner, mock_processor, sample_text_file, output_dir):
    """Test processing a file with both output formats."""
    result = runner.invoke(cli, ['process', sample_text_file, '-o', output_dir, '--format', 'both'])
    assert result.exit_code == 0
    
    # Check if both files exist
    txt_file = Path(output_dir) / "test_output.txt"
    json_file = Path(output_dir) / "test_output.json"
    assert txt_file.exists()
    assert json_file.exists()

def test_estimate_single_file(runner, sample_text_file):
    """Test token estimation for a single file."""
    result = runner.invoke(cli, ['estimate', sample_text_file])
    assert result.exit_code == 0
    assert "Token Usage Estimate" in result.output
    assert "Cost Estimate" in result.output

def test_estimate_directory(runner, tmp_path):
    """Test token estimation for a directory."""
    # Create multiple test files
    (tmp_path / "test1.txt").write_text("Test document 1")
    (tmp_path / "test2.txt").write_text("Test document 2")
    
    result = runner.invoke(cli, ['estimate', str(tmp_path), '--recursive'])
    assert result.exit_code == 0
    assert "Token Usage Estimate" in result.output
    assert "Total Estimated Cost:" in result.output

def test_process_invalid_input(runner, output_dir):
    """Test processing with invalid input file."""
    result = runner.invoke(cli, ['process', 'nonexistent.txt', '-o', output_dir])
    assert result.exit_code != 0
    assert "Error" in result.output or "Invalid" in result.output

def test_process_invalid_format(runner, sample_text_file, output_dir):
    """Test processing with invalid output format."""
    result = runner.invoke(cli, ['process', sample_text_file, '-o', output_dir, '--format', 'invalid'])
    assert result.exit_code != 0
    assert "Invalid value" in result.output

def test_process_with_template(runner, mock_processor, sample_text_file, output_dir):
    """Test processing with a specific template."""
    result = runner.invoke(cli, [
        'process', 
        sample_text_file, 
        '-o', 
        output_dir, 
        '--template', 
        'default'
    ])
    assert result.exit_code == 0

def test_process_with_max_tokens(runner, mock_processor, sample_text_file, output_dir):
    """Test processing with max tokens limit."""
    result = runner.invoke(cli, [
        'process', 
        sample_text_file, 
        '-o', 
        output_dir, 
        '--max-tokens', 
        '100'
    ])
    assert result.exit_code == 0
    mock_processor.return_value.process_text.assert_called_once()
