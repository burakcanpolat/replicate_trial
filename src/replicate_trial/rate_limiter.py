"""
Rate Limiter for API requests.

This module implements a token bucket algorithm for rate limiting API requests.
It helps prevent API quota exhaustion by controlling the rate of requests.
"""

import time
from threading import Lock
from typing import Optional

class RateLimiter:
    """
    Token bucket implementation for rate limiting.
    
    This class implements the token bucket algorithm to control request rates.
    It is thread-safe and supports burst handling.
    
    Attributes:
        rate (float): Number of tokens (requests) per second
        max_tokens (int): Maximum number of tokens that can be accumulated
        tokens (float): Current number of available tokens
        last_update (float): Timestamp of last token update
        lock (Lock): Thread lock for synchronization
    """
    
    def __init__(self, rate: float, max_burst: int = 10):
        """
        Initialize the rate limiter.
        
        Args:
            rate (float): Number of tokens (requests) per second
            max_burst (int): Maximum number of tokens that can be accumulated
        """
        self.rate = rate
        self.max_tokens = max_burst
        self.tokens = max_burst
        self.last_update = time.time()
        self.lock = Lock()
    
    def _add_tokens(self) -> None:
        """
        Add tokens based on elapsed time.
        
        This internal method calculates and adds new tokens based on
        the time elapsed since the last update.
        """
        now = time.time()
        time_passed = now - self.last_update
        new_tokens = time_passed * self.rate
        
        self.tokens = min(self.max_tokens, self.tokens + new_tokens)
        self.last_update = now
    
    def acquire(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """
        Acquire tokens for making a request.
        
        Args:
            tokens (int): Number of tokens to acquire
            timeout (Optional[float]): Maximum time to wait for tokens
        
        Returns:
            bool: True if tokens were acquired, False if timed out
        
        Raises:
            ValueError: If requested tokens exceed max_tokens
        """
        if tokens > self.max_tokens:
            raise ValueError(f"Requested tokens ({tokens}) exceed maximum ({self.max_tokens})")
        
        start_time = time.time()
        
        with self.lock:
            self._add_tokens()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            if timeout is not None:
                if time.time() - start_time >= timeout:
                    return False
            
            return False  # Return False if not enough tokens and no waiting

    def release(self) -> None:
        """
        Release a token back to the bucket.
        
        This method adds one token back to the bucket, up to the maximum limit.
        It is thread-safe and should be called after a request is completed.
        """
        with self.lock:
            self._add_tokens()
            self.tokens = min(self.max_tokens, self.tokens + 1)
