"""
Test module for the rate limiter.

This module contains unit tests for the RateLimiter class to verify
its functionality in controlling API request rates.
"""

import time
import pytest
from replicate_trial.rate_limiter import RateLimiter

def test_rate_limiter_init():
    """Test rate limiter initialization."""
    limiter = RateLimiter(rate=1.0, max_burst=5)
    assert limiter.rate == 1.0
    assert limiter.max_tokens == 5
    assert limiter.tokens == 5

def test_rate_limiter_acquire_success():
    """Test successful token acquisition."""
    limiter = RateLimiter(rate=2.0, max_burst=3)  # 2 tokens per second
    assert limiter.acquire(tokens=1)  # Should succeed immediately
    assert limiter.acquire(tokens=1)  # Should succeed immediately
    assert limiter.acquire(tokens=1)  # Should succeed immediately

def test_rate_limiter_acquire_timeout():
    """Test token acquisition with timeout."""
    limiter = RateLimiter(rate=1.0, max_burst=1)  # 1 token per second
    assert limiter.acquire(tokens=1)  # Should succeed immediately
    
    # Try to acquire another token with a short timeout
    start_time = time.time()
    result = limiter.acquire(tokens=1, timeout=0.1)
    elapsed_time = time.time() - start_time
    
    assert not result  # Should fail due to timeout
    assert elapsed_time < 0.2  # Should not wait longer than timeout

def test_rate_limiter_burst():
    """Test burst handling."""
    limiter = RateLimiter(rate=1.0, max_burst=3)  # 1 token per second, max 3 tokens
    
    # Should be able to use all burst tokens immediately
    assert limiter.acquire(tokens=3)  # Use all tokens
    
    # Should fail to acquire more tokens immediately
    assert not limiter.acquire(tokens=1, timeout=0)

def test_rate_limiter_refill():
    """Test token refill over time."""
    limiter = RateLimiter(rate=2.0, max_burst=2)  # 2 tokens per second
    
    # Use all tokens
    assert limiter.acquire(tokens=2)
    
    # Wait for 1 second (should get 2 new tokens)
    time.sleep(1.0)
    
    # Should be able to acquire tokens again
    assert limiter.acquire(tokens=2)

def test_rate_limiter_invalid_tokens():
    """Test requesting more tokens than maximum."""
    limiter = RateLimiter(rate=1.0, max_burst=5)
    
    with pytest.raises(ValueError):
        limiter.acquire(tokens=6)  # More than max_burst

def test_rate_limiter_parallel_requests():
    """Test rate limiter with parallel requests."""
    import threading
    import time

    limiter = RateLimiter(rate=10.0, max_burst=10)
    success_count = 0
    thread_count = 20
    lock = threading.Lock()

    def make_request():
        nonlocal success_count
        if limiter.acquire(tokens=1, timeout=0.1):  # Shorter timeout
            with lock:
                success_count += 1

    # Wait for rate limiter to initialize
    time.sleep(0.1)

    threads = []
    for _ in range(thread_count):
        thread = threading.Thread(target=make_request)
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    # Should only succeed for max_burst number of requests
    assert success_count <= 10
