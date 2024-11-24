#!/usr/bin/env python3
"""
Command Line Interface for Text Processing using Llama-2.

This module provides a command-line interface for processing text files using Meta's
Llama-2 model through the Replicate API. It supports various features like cost
estimation, different output formats, and recursive directory processing.

Key Features:
    - Text processing with Llama-2
    - Cost estimation before processing
    - Multiple output formats (JSON/TXT)
    - Recursive directory processing
    - Progress tracking with rich output

Example Usage:
    Process a single file:
    ```bash
    python cli.py process input.txt -o output_dir
    ```
    
    Estimate cost for a directory:
    ```bash
    python cli.py estimate input_dir --recursive
    ```

Dependencies:
    - click: For CLI argument parsing
    - rich: For enhanced console output
    - python-dotenv: For environment variables
"""

import os
import sys
import json
import textwrap
from pathlib import Path
import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from dotenv import load_dotenv
from typing import Optional

from replicate_trial.replicate_processor import ReplicateProcessor
from replicate_trial.token_counter import TokenCounter
from replicate_trial.prompt_templates import TEMPLATES

# Initialize console for rich output
console = Console()

def process_file(
    processor: ReplicateProcessor,
    counter: TokenCounter,
    input_path: Path,
    output_dir: Path,
    template_key: str,
    format: str,
    dry_run: bool,
    max_tokens: Optional[int]
) -> None:
    """
    Process a single text file using Llama-2.
    
    This function:
    1. Reads the input file
    2. Estimates processing cost
    3. If not dry_run, processes the text
    4. Saves the results in specified format(s)
    
    Args:
        processor (ReplicateProcessor): Instance of text processor
        counter (TokenCounter): Instance of token counter
        input_path (Path): Path to input file
        output_dir (Path): Directory to save output
        template_key (str): Template to use for processing
        format (str): Output format (json/txt/both)
        dry_run (bool): If True, only show cost estimate
        max_tokens (Optional[int]): Maximum output tokens
        
    Raises:
        IOError: If there are issues reading/writing files
        ReplicateAPIError: If there are issues with the API
    """
    # Read input file
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Get cost estimate
    estimate = counter.estimate_cost(text, max_tokens)
    console.print(Panel(
        counter.format_cost_estimate(estimate),
        title=f"Cost Estimate for {input_path.name}",
        border_style="blue"
    ))
    
    if dry_run:
        return
    
    # Process text
    try:
        result = processor.process_text(text, template_key=template_key)
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save outputs based on format
        if format in ['json', 'both']:
            json_path = output_dir / f"{input_path.stem}_output.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
        
        if format in ['txt', 'both']:
            txt_path = output_dir / f"{input_path.stem}_output.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                # Write metadata section
                f.write("METADATA\n")
                f.write("========\n\n")
                
                f.write("Summary:\n")
                f.write("--------\n")
                f.write(textwrap.fill(result['metadata']['summary'], width=100) + "\n\n")
                
                f.write("Tags:\n")
                f.write("-----\n")
                f.write(", ".join(result['metadata']['tags']) + "\n\n")
                
                f.write("Key Points:\n")
                f.write("-----------\n")
                for point in result['metadata']['key_points']:
                    f.write(f"* {textwrap.fill(point, width=95, initial_indent='', subsequent_indent='  ')}\n")
                f.write("\n")
                
                # Write formatted text section
                f.write("FORMATTED TEXT\n")
                f.write("==============\n\n")
                f.write(result['formatted_text'])
    
    except Exception as e:
        console.print(f"[red]Error processing {input_path}: {str(e)}[/red]")

@click.group()
def cli():
    """Text processing CLI using Llama-2 through Replicate API."""
    # Load environment variables
    load_dotenv()
    
    # Check for API token
    if not os.environ.get("REPLICATE_API_TOKEN"):
        console.print("[red]Error: REPLICATE_API_TOKEN environment variable not set[/red]")
        sys.exit(1)

@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--output-dir', '-o', type=click.Path(), default='output',
              help='Output directory for processed files')
@click.option('--format', '-f', type=click.Choice(['json', 'txt', 'both']),
              default='both', help='Output format')
@click.option('--template', '-t', type=click.Choice(list(TEMPLATES.keys())),
              default='default', help='Template to use for processing')
@click.option('--recursive', '-r', is_flag=True, help='Process files in subdirectories')
@click.option('--dry-run', is_flag=True, help='Show what would be done without doing it')
@click.option('--max-tokens', type=int, help='Maximum output tokens')
def process(
    input_path: str,
    output_dir: str,
    format: str,
    template: str,
    recursive: bool,
    dry_run: bool,
    max_tokens: Optional[int]
):
    """
    Process text file(s) using Llama-2.
    
    This command:
    1. Takes input file(s) and processes them using Llama-2
    2. Saves the results in specified format(s)
    3. Supports recursive directory processing
    4. Can do a dry run to show cost estimates
    
    Args:
        input_path: Path to input file or directory
        output_dir: Directory to save processed files
        format: Output format (json/txt/both)
        template: Template to use for processing
        recursive: Process files in subdirectories
        dry_run: Only show cost estimates
        max_tokens: Maximum output tokens
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    
    # Initialize processor and counter
    processor = ReplicateProcessor()
    counter = TokenCounter()
    
    if input_path.is_file():
        # Process single file
        process_file(processor, counter, input_path, output_dir, template, format, dry_run, max_tokens)
    else:
        # Process directory
        pattern = '**/*' if recursive else '*'
        files = list(input_path.glob(pattern))
        text_files = [f for f in files if f.is_file() and f.suffix.lower() in ['.txt', '.md']]
        
        if not text_files:
            console.print("[yellow]No text files found to process[/yellow]")
            return
        
        with Progress() as progress:
            task = progress.add_task("[cyan]Processing files...", total=len(text_files))
            
            for file in text_files:
                process_file(processor, counter, file, output_dir, template, format, dry_run, max_tokens)
                progress.advance(task)

@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--recursive', '-r', is_flag=True, help='Estimate cost for files in subdirectories')
def estimate(input_path: str, recursive: bool):
    """
    Estimate token count and cost for text file(s).
    
    This command:
    1. Counts tokens in input file(s)
    2. Estimates processing cost using current Replicate rates
    3. Can process entire directories recursively
    
    Args:
        input_path: Path to input file or directory
        recursive: Process files in subdirectories
    """
    input_path = Path(input_path)
    counter = TokenCounter()
    
    def estimate_file(file_path: Path) -> None:
        """Helper function to estimate cost for a single file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        estimate = counter.estimate_cost(text)
        console.print(Panel(
            counter.format_cost_estimate(estimate),
            title=str(file_path),
            border_style="blue"
        ))
    
    if input_path.is_file():
        estimate_file(input_path)
    else:
        pattern = '**/*' if recursive else '*'
        files = list(input_path.glob(pattern))
        text_files = [f for f in files if f.is_file() and f.suffix.lower() in ['.txt', '.md']]
        
        if not text_files:
            console.print("[yellow]No text files found to estimate[/yellow]")
            return
        
        total_cost = 0
        for file in text_files:
            with open(file, 'r', encoding='utf-8') as f:
                text = f.read()
            estimate = counter.estimate_cost(text)
            total_cost += estimate['total_cost']
            console.print(Panel(
                counter.format_cost_estimate(estimate),
                title=str(file),
                border_style="blue"
            ))
        
        console.print(Panel(
            f"Total Estimated Cost: ${total_cost:.6f}",
            title="Summary",
            border_style="green"
        ))

if __name__ == '__main__':
    cli()
