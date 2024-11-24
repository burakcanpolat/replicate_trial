"""
Text Processing Module using Replicate API with Llama-2.

This module provides a high-level interface for text processing tasks using Meta's Llama-2 model
through Replicate's API. It handles authentication, retries, timeouts, template management,
and rate limiting automatically.

Key Features:
    - Automatic API token verification
    - Configurable timeout settings
    - Exponential backoff retry mechanism
    - Template-based text processing
    - JSON response parsing
    - File output formatting

Example:
    ```python
    from replicate_processor import ReplicateProcessor
    
    # Initialize processor
    processor = ReplicateProcessor()
    
    # Process text
    result = processor.process_text(
        "Your text here",
        template_key="default"
    )
    
    # Save results
    processor.save_processed_text(result, "output.txt")
    ```

Dependencies:
    - replicate: For API interaction
    - python-dotenv: For environment variable management
    - requests: For HTTP requests
"""

import os
import json
import time
import re
from typing import Dict, Any, Optional
import requests
import replicate
from dotenv import load_dotenv
from replicate_trial.prompt_templates import TEMPLATES
from replicate_trial.rate_limiter import RateLimiter

class ReplicateAPIError(Exception):
    """
    Custom exception for Replicate API errors.
    
    This exception is raised when there are issues with the Replicate API,
    such as authentication failures, timeouts, or API response errors.
    
    Attributes:
        message (str): The error message
    """
    pass

class ReplicateProcessor:
    """
    A class to handle text processing using Llama-2 through Replicate's API.
    
    This class provides a high-level interface for text processing tasks using
    Meta's Llama-2 model via Replicate's API. It handles authentication, retries,
    timeouts, template management, and rate limiting automatically.
    
    Attributes:
        api_token (str): The Replicate API token from environment variables
        model_version (str): The Llama-2 7B model version ID
        timeout (int): Timeout duration for API calls in seconds
        rate_limiter (RateLimiter): Rate limiter for API requests
    """
    
    def __init__(self, 
                 model_version: str = "meta/llama-2-7b-chat:13c3cdee13ee059ab779f0291d29054dab00a47dad8261375654de5540165fb0",
                 timeout: int = 300,
                 requests_per_minute: float = 60.0):
        """
        Initialize the processor with Replicate API token and Llama-2 model.
        
        This method:
        1. Loads the API token from environment variables
        2. Verifies the token's validity
        3. Configures the model, timeout, and rate limiting settings
        
        Args:
            model_version (str): The Llama-2 7B model version ID
            timeout (int): Timeout in seconds for API calls. Defaults to 300 seconds.
            requests_per_minute (float): Maximum number of requests per minute. Defaults to 60.
            
        Raises:
            ValueError: If REPLICATE_API_TOKEN is not found in environment variables
            ReplicateAPIError: If the API token verification fails
        """
        # Load environment variables
        load_dotenv()
        
        self.api_token = os.getenv("REPLICATE_API_TOKEN")
        if not self.api_token:
            raise ValueError("REPLICATE_API_TOKEN not found in environment variables")
        
        self.model_version = model_version
        self.timeout = timeout
        
        # Initialize rate limiter (convert requests per minute to requests per second)
        self.rate_limiter = RateLimiter(rate=requests_per_minute / 60.0, max_burst=10)
        
        # Verify API token
        self._verify_api_token()

    def _verify_api_token(self) -> bool:
        """
        Verify that the API token is valid by making a test request.
        
        This method:
        1. Makes a test request to the Replicate API account endpoint
        2. Verifies that the response is successful
        3. Handles any connection or authentication errors
        
        Returns:
            bool: True if token is valid and verification succeeds
            
        Raises:
            ReplicateAPIError: If token is invalid, API is unreachable, or any other
                             request error occurs
        """
        try:
            headers = {"Authorization": f"Token {self.api_token}"}
            response = requests.get(
                "https://api.replicate.com/v1/account",
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            raise ReplicateAPIError(f"Failed to verify API token: {str(e)}")

    def _clean_json_string(self, json_str: str) -> str:
        """
        Clean and format JSON string for proper parsing.
        
        Args:
            json_str (str): Raw JSON string to clean
            
        Returns:
            str: Cleaned JSON string
        """
        # Find the first { and last }
        start_idx = json_str.find('{')
        end_idx = json_str.rfind('}')
        
        if start_idx == -1 or end_idx == -1:
            raise ValueError("No valid JSON object found in string")
            
        # Extract the JSON portion
        json_str = json_str[start_idx:end_idx + 1]
        
        try:
            # First attempt: Try to parse as is
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                # Second attempt: Try to extract metadata and text separately
                metadata_match = re.search(r'"metadata":\s*({[^}]+})', json_str)
                text_start = json_str.find('"formatted_text"') + len('"formatted_text"')
                text_end = json_str.rfind('}')
                
                if not metadata_match or text_start == -1:
                    raise ValueError("Could not find metadata or formatted_text")
                
                metadata = json.loads(metadata_match.group(1))
                
                # Extract the text portion and clean it
                text_portion = json_str[text_start:text_end].strip()
                if text_portion.startswith(':'):
                    text_portion = text_portion[1:]
                if text_portion.endswith(','):
                    text_portion = text_portion[:-1]
                    
                # Clean the text portion
                text_portion = text_portion.strip().strip('"').strip("'")
                
                # Create a clean dictionary
                clean_dict = {
                    "metadata": metadata,
                    "formatted_text": text_portion
                }
                
                return json.dumps(clean_dict, ensure_ascii=False)
            except Exception as e:
                print(f"Error in second attempt: {str(e)}")
                
            try:
                # Third attempt: Try a more aggressive cleaning
                # Remove all control characters except newlines
                json_str = ''.join(char if char >= ' ' else '\\n' if char == '\n' else '' for char in json_str)
                
                # Fix common JSON issues
                json_str = (json_str
                          .replace("'", '"')  # Replace single quotes
                          .replace('True', 'true')
                          .replace('False', 'false')
                          .replace('None', 'null'))
                
                return json_str
            except Exception as e:
                print(f"Error in third attempt: {str(e)}")
                raise ValueError("Unable to clean JSON string")

    def _chunk_text(self, text: str, max_chunk_size: int = 1500) -> list[str]:
        """
        Split text into chunks that respect token limits.
        Uses a simple approximation of 4 characters per token.
        
        Args:
            text (str): Text to split into chunks
            max_chunk_size (int): Maximum chunk size in tokens
            
        Returns:
            list[str]: List of text chunks
        """
        # Approximate chars per token (4 is a conservative estimate)
        chars_per_token = 4
        max_chars = max_chunk_size * chars_per_token
        
        # If text is short enough, return as is
        if len(text) <= max_chars:
            return [text]
            
        chunks = []
        current_pos = 0
        
        while current_pos < len(text):
            # Find a good breaking point
            end_pos = current_pos + max_chars
            if end_pos >= len(text):
                chunks.append(text[current_pos:])
                break
                
            # Try to break at sentence end
            sentence_end = text.rfind('. ', current_pos, end_pos)
            if sentence_end != -1:
                end_pos = sentence_end + 1
            else:
                # Try to break at paragraph
                para_end = text.rfind('\n', current_pos, end_pos)
                if para_end != -1:
                    end_pos = para_end + 1
                else:
                    # Try to break at word boundary
                    word_end = text.rfind(' ', current_pos, end_pos)
                    if word_end != -1:
                        end_pos = word_end + 1
            
            chunks.append(text[current_pos:end_pos])
            current_pos = end_pos
            
        return chunks

    def process_text(self, text: str, template_key: str = "default", max_retries: int = 3) -> Dict[str, Any]:
        """
        Process text using Llama-2 model with rate limiting.

        This method:
        1. Acquires a rate limiting token
        2. Applies the specified template
        3. Makes the API call with retries and timeout
        4. Returns the processed result

        Args:
            text (str): The text to process
            template_key (str): The template to use for processing
            max_retries (int): Maximum number of retries for failed API calls. Defaults to 3.

        Returns:
            Dict[str, Any]: The processed result containing metadata and formatted text

        Raises:
            ReplicateAPIError: If API call fails or rate limit is exceeded
            ValueError: If template_key is invalid
            TimeoutError: If the API call times out
        """
        if not self.rate_limiter.acquire(timeout=self.timeout):
            raise ReplicateAPIError("Rate limit exceeded. Please try again later.")

        try:
            # Get template
            template = TEMPLATES.get(template_key)
            if not template:
                raise ValueError(f"Template '{template_key}' not found")

            chunks = self._chunk_text(text)
            results = []

            for chunk in chunks:
                retry_count = 0
                last_error = None

                while retry_count < max_retries:
                    try:
                        # Run the Llama-2 model with streaming
                        output_stream = replicate.stream(
                            self.model_version,
                            input={
                                "prompt": f"{template['system_prompt']}\n\nText to process:\n\n{chunk}",
                                "temperature": 0.1,
                                "top_p": 0.9,
                                "max_tokens": 4096,
                                "min_tokens": 1,
                                "repetition_penalty": 1.1,
                                "system_prompt": "You are a helpful assistant that formats and analyzes text."
                            }
                        )

                        # Collect streaming output with timeout
                        full_response = ""
                        start_time = time.time()
                        
                        for event in output_stream:
                            if time.time() - start_time > self.timeout:
                                raise TimeoutError("API call exceeded timeout")
                            full_response += str(event)

                        # Clean and parse the JSON response
                        cleaned_response = (full_response
                                             .replace('\n', '')
                                             .replace("'", '"')
                                             .replace('True', 'true')
                                             .replace('False', 'false')
                                             .replace('None', 'null'))
                        
                        try:
                            response_data = json.loads(cleaned_response)
                            
                            # Extract metadata and formatted text
                            metadata = response_data.get('metadata', {})
                            formatted_text = response_data.get('formatted_text', '')
                            
                            # Clean the formatted text
                            formatted_text = re.sub(r'[{}\[\]"`]', '', formatted_text)
                            formatted_text = re.sub(r'metadata:|formatted_text:|summary:|tags:|key_points:', '', formatted_text)
                            formatted_text = re.sub(r'^.*?(Here is|The following|This is|Your).*?:', '', formatted_text)
                            formatted_text = formatted_text.strip()
                            
                            # Add paragraph breaks
                            formatted_text = re.sub(r'(?<=[.!?])\s+(?=[A-Z])', '\n\n', formatted_text)
                            
                            # Validate metadata structure
                            if not isinstance(metadata, dict):
                                metadata = {}
                            if 'summary' not in metadata:
                                metadata['summary'] = ''
                            if 'tags' not in metadata:
                                metadata['tags'] = []
                            if 'key_points' not in metadata:
                                metadata['key_points'] = []
                            
                            results.append({
                                'metadata': metadata,
                                'formatted_text': formatted_text or chunk
                            })
                            break
                        except json.JSONDecodeError:
                            # If JSON parsing fails, try to extract content directly
                            # Extract metadata and formatted text using regex
                            metadata_match = re.search(r'metadata"?:\s*({[^}]+})', full_response)
                            text_match = re.search(r'formatted_text"?:\s*"([^"]+)"', full_response)
                            
                            metadata = {}
                            if metadata_match:
                                try:
                                    metadata = json.loads(metadata_match.group(1).replace("'", '"'))
                                except:
                                    pass
                            
                            formatted_text = text_match.group(1) if text_match else full_response
                            formatted_text = re.sub(r'[{}\[\]"`]', '', formatted_text)
                            formatted_text = re.sub(r'metadata:|formatted_text:|summary:|tags:|key_points:', '', formatted_text)
                            formatted_text = re.sub(r'^.*?(Here is|The following|This is|Your).*?:', '', formatted_text)
                            formatted_text = formatted_text.strip()
                            
                            # Add paragraph breaks
                            formatted_text = re.sub(r'(?<=[.!?])\s+(?=[A-Z])', '\n\n', formatted_text)
                            
                            # Validate metadata structure
                            if not isinstance(metadata, dict):
                                metadata = {}
                            if 'summary' not in metadata:
                                metadata['summary'] = ''
                            if 'tags' not in metadata:
                                metadata['tags'] = []
                            if 'key_points' not in metadata:
                                metadata['key_points'] = []
                            
                            results.append({
                                'metadata': metadata,
                                'formatted_text': formatted_text or chunk
                            })
                            break
                    except Exception as e:
                        last_error = str(e)
                        retry_count += 1
                        if retry_count < max_retries:
                            time.sleep(min(2 ** retry_count, 5))
                            continue
                        raise ReplicateAPIError(f"Failed to process text after {max_retries} retries: {last_error}")

            # Combine results from chunks
            if not results:
                raise ReplicateAPIError("No valid results obtained from processing")

            return {
                'metadata': results[0]['metadata'],
                'formatted_text': ' '.join(result['formatted_text'] for result in results)
            }

        except Exception as e:
            raise ReplicateAPIError(f"Error processing text: {str(e)}")
        finally:
            self.rate_limiter.release()

    def save_processed_text(self, processed_result: Dict[str, Any], output_file: str):
        """
        Save the processed text to a file.

        This method writes the processed text and analysis to a formatted text file.
        The output includes sections for:
        - Metadata (summary, tags, key points)
        - Formatted text with proper spacing and structure

        Args:
            processed_result (Dict[str, Any]): The processed text and analysis containing:
                - formatted_text (str): The processed text
                - metadata (dict, optional):
                    - summary (str): Text summary
                    - tags (List[str]): Relevant tags
                    - key_points (List[str]): Key points from the text
            output_file (str): Path to save the output file

        Raises:
            IOError: If there are issues writing to the output file
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write metadata section
            f.write("METADATA\n")
            f.write("========\n\n")
            
            # Get metadata if it exists
            metadata = processed_result.get('metadata', {})
            
            # Write summary if available
            if 'summary' in metadata and metadata['summary']:
                f.write("Summary:\n")
                f.write("--------\n")
                f.write(metadata['summary'] + "\n\n")

            # Write tags if available
            if 'tags' in metadata and metadata['tags']:
                f.write("Tags:\n")
                f.write("-----\n")
                f.write(", ".join(metadata['tags']) + "\n\n")

            # Write key points if available
            if 'key_points' in metadata and metadata['key_points']:
                f.write("Key Points:\n")
                f.write("-----------\n")
                for point in metadata['key_points']:
                    f.write(f"* {point}\n")
                f.write("\n")

            # Write formatted text with proper spacing
            f.write("FORMATTED TEXT\n")
            f.write("==============\n\n")
            formatted_text = processed_result.get('formatted_text', '')
            
            # Clean any remaining analysis text
            formatted_text = re.sub(r'Based on the provided text.*?follows:', '', formatted_text, flags=re.DOTALL)
            formatted_text = re.sub(r'Summary:.*?Key Points:', '', formatted_text, flags=re.DOTALL)
            formatted_text = re.sub(r'Here is the formatted text.*?:', '', formatted_text, flags=re.DOTALL)
            
            # Format paragraphs
            # Split into sentences
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', formatted_text)
            
            # Group sentences into paragraphs based on topic similarity
            paragraphs = []
            current_paragraph = []
            
            for sentence in sentences:
                if not sentence.strip():
                    continue
                    
                # Start a new paragraph if:
                # 1. Current paragraph is empty
                # 2. Sentence starts with a topic-changing phrase
                # 3. Current paragraph is getting too long (> 5 sentences)
                topic_changes = ['so ', 'now ', 'then ', 'but ', 'and so ', 'first ', 'second ', 'finally ', 'next ']
                if (not current_paragraph or
                    any(sentence.lower().startswith(phrase) for phrase in topic_changes) or
                    len(current_paragraph) > 5):
                    if current_paragraph:
                        paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = [sentence]
                else:
                    current_paragraph.append(sentence)
            
            # Add the last paragraph
            if current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
            
            # Join paragraphs with double newlines
            formatted_text = '\n\n'.join(paragraphs)
            formatted_text = formatted_text.strip()
            
            f.write(formatted_text)
