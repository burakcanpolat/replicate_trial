# Replicate Trial

A Python package for processing and formatting raw transcripts using the Replicate API and Llama-2 model. The package focuses on preserving original content while improving readability through proper formatting, punctuation, and paragraph structure.

## Features

- Transcript processing using Llama-2 model via Replicate API
- Content preservation with minimal formatting changes
- Metadata extraction (summary, tags, key points)
- Token counting and cost estimation
- Multiple output formats (metadata, text, or both)
- Progress tracking and rich console output
- Rate limiting for API calls

## Project Structure

```
replicate_trial/
├── src/
│   └── replicate_trial/
│       ├── __init__.py
│       ├── cli.py              # Command-line interface
│       ├── replicate_processor.py  # Core processing logic
│       ├── rate_limiter.py     # API rate limiting
│       └── prompt_templates.py  # Llama-2 prompt templates
├── tests/
│   ├── test_replicate.py
│   └── test_data/
│       ├── final_trial.txt
│       └── small_test.txt
├── setup.py                   # Package installation
├── requirements.txt           # Development dependencies
├── pytest.ini                # Test configuration
├── LICENSE
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/replicate_trial.git
cd replicate_trial
```

2. Create and activate a virtual environment:
```bash
python -m venv rep
source rep/bin/activate  # On Windows, use: rep\Scripts\activate
```

3. Install the package in development mode:
```bash
pip install -e .
```

4. Set up your Replicate API token:
```bash
echo "REPLICATE_API_TOKEN=your_token_here" > .env
```

## Usage

The package provides a command-line interface for text processing:

```bash
# Process text from a file
python -m replicate_trial.cli process input.txt -o output --format both

# Process with custom max tokens
python -m replicate_trial.cli process input.txt -o output --max-tokens 4096

# Get help
python -m replicate_trial.cli --help
```

## Development Roadmap

### Current Issues
- [ ] Improve text cleaning to better handle raw transcripts
- [ ] Fix metadata extraction from API response
- [ ] Enhance paragraph formatting logic
- [ ] Add better error handling for API failures

### Planned Features
1. Text Processing
   - [ ] Add support for different text styles (academic, technical, casual)
   - [ ] Implement custom formatting rules
   - [ ] Add support for different output formats (markdown, HTML)
   - [ ] Improve sentence boundary detection

2. API Integration
   - [ ] Add support for multiple LLM providers
   - [ ] Implement retry logic for API failures
   - [ ] Add caching for processed results
   - [ ] Optimize token usage

3. Testing & Documentation
   - [ ] Add comprehensive test suite
   - [ ] Add API documentation
   - [ ] Create usage examples
   - [ ] Add performance benchmarks

4. User Experience
   - [ ] Add progress bars for long processes
   - [ ] Implement batch processing
   - [ ] Add configuration file support
   - [ ] Create interactive mode

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
