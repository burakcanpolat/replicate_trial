"""Module for estimating token counts and costs for text processing."""
import tiktoken
from typing import Dict, Union, Optional
from enum import Enum

class ModelProvider(Enum):
    """Enum for model providers"""
    META = "Meta"
    IBM = "IBM"
    MISTRAL = "Mistral AI"

# Model configurations with their specifications and costs
MODEL_CONFIGS = {
    # IBM Granite Models
    "granite-20b-code-instruct-8k": {
        "id": "ibm/granite-20b-code-instruct-8k",
        "provider": ModelProvider.IBM,
        "cost_per_1m_input_tokens": 0.100,
        "cost_per_1m_output_tokens": 0.500,
        "max_input_tokens": 8192,
        "max_output_tokens": 8192,
        "description": "IBM Granite 20B code instruct model with 8K context"
    },
    "granite-3.0-2b-instruct": {
        "id": "ibm/granite-3.0-2b-instruct",
        "provider": ModelProvider.IBM,
        "cost_per_1m_input_tokens": 0.030,
        "cost_per_1m_output_tokens": 0.250,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "description": "IBM Granite 3.0 2B instruct model"
    },
    "granite-3.0-8b-instruct": {
        "id": "ibm/granite-3.0-8b-instruct",
        "provider": ModelProvider.IBM,
        "cost_per_1m_input_tokens": 0.050,
        "cost_per_1m_output_tokens": 0.250,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "description": "IBM Granite 3.0 8B instruct model"
    },
    
    # Meta Llama 2 Models
    "llama-2-7b": {
        "id": "meta/llama-2-7b",
        "provider": ModelProvider.META,
        "cost_per_1m_input_tokens": 0.050,
        "cost_per_1m_output_tokens": 0.250,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "description": "Llama 2 7B base model"
    },
    "llama-2-7b-chat": {
        "id": "meta/llama-2-7b-chat",
        "provider": ModelProvider.META,
        "cost_per_1m_input_tokens": 0.050,
        "cost_per_1m_output_tokens": 0.250,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "description": "Llama 2 7B chat model"
    },
    "llama-2-13b": {
        "id": "meta/llama-2-13b",
        "provider": ModelProvider.META,
        "cost_per_1m_input_tokens": 0.100,
        "cost_per_1m_output_tokens": 0.500,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "description": "Llama 2 13B base model"
    },
    "llama-2-70b": {
        "id": "meta/llama-2-70b",
        "provider": ModelProvider.META,
        "cost_per_1m_input_tokens": 0.650,
        "cost_per_1m_output_tokens": 2.750,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "description": "Llama 2 70B base model"
    },
    
    # Meta Llama 3 Models
    "meta-llama-3-8b": {
        "id": "meta/llama-3-8b",
        "provider": ModelProvider.META,
        "cost_per_1m_input_tokens": 0.050,
        "cost_per_1m_output_tokens": 0.250,
        "max_input_tokens": 8192,
        "max_output_tokens": 8192,
        "description": "Llama 3 8B base model"
    },
    "meta-llama-3-70b": {
        "id": "meta/llama-3-70b",
        "provider": ModelProvider.META,
        "cost_per_1m_input_tokens": 0.650,
        "cost_per_1m_output_tokens": 2.750,
        "max_input_tokens": 8192,
        "max_output_tokens": 8192,
        "description": "Llama 3 70B base model"
    },
    "meta-llama-3.1-405b-instruct": {
        "id": "meta/llama-3.1-405b-instruct",
        "provider": ModelProvider.META,
        "cost_per_1m_input_tokens": 9.500,
        "cost_per_1m_output_tokens": 9.500,
        "max_input_tokens": 8192,
        "max_output_tokens": 8192,
        "description": "Llama 3.1 405B instruct model"
    },
    
    # Mistral AI Models
    "mistral-7b-v0.1": {
        "id": "mistralai/mistral-7b-v0.1",
        "provider": ModelProvider.MISTRAL,
        "cost_per_1m_input_tokens": 0.050,
        "cost_per_1m_output_tokens": 0.250,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "description": "Mistral 7B v0.1 base model"
    },
    "mistral-7b-instruct-v0.2": {
        "id": "mistralai/mistral-7b-instruct-v0.2",
        "provider": ModelProvider.MISTRAL,
        "cost_per_1m_input_tokens": 0.050,
        "cost_per_1m_output_tokens": 0.250,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "description": "Mistral 7B v0.2 instruct model"
    },
    "mixtral-8x7b-instruct-v0.1": {
        "id": "mistralai/mixtral-8x7b-instruct-v0.1",
        "provider": ModelProvider.MISTRAL,
        "cost_per_1m_input_tokens": 0.300,
        "cost_per_1m_output_tokens": 1.000,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "description": "Mixtral 8x7B v0.1 instruct model"
    }
}

class TokenCounter:
    """
    Class to handle token counting and cost estimation.
    
    This class provides functionality to:
    1. Count tokens in input text
    2. Estimate output token usage
    3. Calculate processing costs
    4. Format cost estimates for display
    
    While it can estimate costs for various models, text processing
    is currently implemented only for Llama-2.
    
    Attributes:
        model_name (str): Name of the model for cost estimation
        model_config (dict): Configuration for the selected model
        encoder: TikToken encoder instance
    """
    
    def __init__(self, model_name: str = "llama-2-7b-chat"):
        """
        Initialize the token counter with a specific model configuration.
        
        Args:
            model_name (str): Name of the model to use for cost estimation.
                            Defaults to "llama-2-7b-chat".
                            
        Raises:
            ValueError: If the specified model is not found in MODEL_CONFIGS
        """
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Model {model_name} not found. Available models: {list(MODEL_CONFIGS.keys())}")
        
        self.model_name = model_name
        self.model_config = MODEL_CONFIGS[model_name]
        
        # Initialize tokenizer
        self.encoder = tiktoken.encoding_for_model("gpt-4")  # Using gpt-4 tokenizer as approximation
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the input text.
        
        Args:
            text (str): The input text to count tokens for
            
        Returns:
            int: Number of tokens in the text
        """
        return len(self.encoder.encode(text))
    
    def estimate_output_tokens(self, input_tokens: int) -> int:
        """
        Estimate the number of output tokens based on input length.
        
        This is a simple heuristic that assumes the output will be
        approximately twice the length of the input.
        
        Args:
            input_tokens (int): Number of input tokens
            
        Returns:
            int: Estimated number of output tokens
        """
        estimated = min(input_tokens * 2, self.model_config["max_output_tokens"])
        return estimated
    
    def estimate_cost(self, text: str, max_output_tokens: Optional[int] = None) -> Dict[str, Union[int, float, str]]:
        """
        Estimate the cost of processing the input text.
        
        This method:
        1. Counts input tokens
        2. Estimates output tokens
        3. Calculates costs based on the model's pricing
        
        Args:
            text (str): Input text to estimate cost for
            max_output_tokens (int, optional): Maximum number of output tokens
            
        Returns:
            Dict[str, Union[int, float, str]]: Cost estimate containing:
                - model_name (str): Name of the model
                - model_id (str): Model ID
                - provider (str): Model provider name
                - input_tokens (int): Number of input tokens
                - estimated_output_tokens (int): Estimated number of output tokens
                - input_cost (float): Cost for input tokens
                - output_cost (float): Estimated cost for output tokens
                - total_cost (float): Total estimated cost
        """
        # Count input tokens
        input_tokens = self.count_tokens(text)
        
        if input_tokens > self.model_config["max_input_tokens"]:
            raise ValueError(
                f"Input text too long ({input_tokens} tokens). "
                f"Maximum allowed is {self.model_config['max_input_tokens']} tokens."
            )
        
        # Estimate output tokens (capped by max_output_tokens or model limit)
        estimated_output_tokens = min(
            max_output_tokens if max_output_tokens else self.estimate_output_tokens(input_tokens),
            self.model_config["max_output_tokens"]
        )
        
        # Calculate costs (convert from per 1M tokens to actual cost)
        input_cost = (input_tokens / 1_000_000) * self.model_config["cost_per_1m_input_tokens"]
        output_cost = (estimated_output_tokens / 1_000_000) * self.model_config["cost_per_1m_output_tokens"]
        total_cost = input_cost + output_cost
        
        return {
            "model_name": self.model_name,
            "model_id": self.model_config["id"],
            "provider": self.model_config["provider"].value,
            "description": self.model_config["description"],
            "input_tokens": input_tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total_cost": round(total_cost, 6)
        }
    
    def format_cost_estimate(self, estimate: Dict[str, Union[int, float, str]]) -> str:
        """
        Format a cost estimate into a human-readable string.
        
        Args:
            estimate (Dict[str, Union[int, float, str]]): Cost estimate from estimate_cost()
            
        Returns:
            str: Formatted string containing the cost estimate details
        """
        return (
            f"Model Information:\n"
            f"  Provider: {estimate['provider']}\n"
            f"  Model: {estimate['model_name']}\n"
            f"  Description: {estimate['description']}\n"
            f"  Model ID: {estimate['model_id']}\n"
            f"\nToken Usage Estimate:\n"
            f"  Input Tokens: {estimate['input_tokens']:,}\n"
            f"  Estimated Output Tokens: {estimate['estimated_output_tokens']:,}\n"
            f"\nCost Estimate:\n"
            f"  Input Cost: ${estimate['input_cost']:.6f}\n"
            f"  Output Cost: ${estimate['output_cost']:.6f}\n"
            f"  Total Cost: ${estimate['total_cost']:.6f}"
        )
    
    @staticmethod
    def list_available_models(provider: Optional[ModelProvider] = None) -> str:
        """
        Return a formatted string listing available models, optionally filtered by provider.
        
        Args:
            provider (Optional[ModelProvider]): Optional ModelProvider enum to filter models by provider
        """
        models = MODEL_CONFIGS.items()
        if provider:
            models = [(name, config) for name, config in models 
                     if config["provider"] == provider]
        
        return "\n\n".join(
            f"{model_name}:\n"
            f"  Provider: {config['provider'].value}\n"
            f"  Description: {config['description']}\n"
            f"  Max Input Tokens: {config['max_input_tokens']:,}\n"
            f"  Cost per 1M input tokens: ${config['cost_per_1m_input_tokens']:.3f}\n"
            f"  Cost per 1M output tokens: ${config['cost_per_1m_output_tokens']:.3f}"
            for model_name, config in sorted(models, key=lambda x: (x[1]['provider'].value, x[0]))
        )
