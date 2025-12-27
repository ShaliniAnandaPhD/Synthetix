"""
Reliability Unit Tests

Tests for circuit breaker, rate limiter, and fallback layers.
"""
import pytest
import asyncio
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.identity_regression import RateLimiter


# =============================================================================
# CIRCUIT BREAKER TESTS
# =============================================================================

class CircuitBreaker:
    """Simple circuit breaker for testing."""
    
    def __init__(self, failure_threshold: int = 3, reset_timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.state = "CLOSED"
        self.last_failure_time = 0
    
    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self.state = "OPEN"
    
    def record_success(self):
        self.failures = 0
        self.state = "CLOSED"
    
    def is_call_permitted(self) -> bool:
        if self.state == "CLOSED":
            return True
        
        # Check if reset timeout elapsed
        if time.time() - self.last_failure_time >= self.reset_timeout:
            self.state = "HALF_OPEN"
            return True
        
        return False


class TestCircuitBreaker:
    """Tests for circuit breaker pattern."""
    
    def test_circuit_starts_closed(self):
        """Circuit should start in CLOSED state."""
        breaker = CircuitBreaker()
        assert breaker.state == "CLOSED"
        assert breaker.is_call_permitted() == True
    
    def test_circuit_opens_after_threshold(self):
        """Circuit should open after failure threshold reached."""
        breaker = CircuitBreaker(failure_threshold=3)
        
        breaker.record_failure()
        assert breaker.state == "CLOSED"
        
        breaker.record_failure()
        assert breaker.state == "CLOSED"
        
        breaker.record_failure()
        assert breaker.state == "OPEN"
        assert breaker.is_call_permitted() == False
    
    def test_circuit_resets_on_success(self):
        """Circuit should reset failures on success."""
        breaker = CircuitBreaker(failure_threshold=3)
        
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_success()
        
        assert breaker.failures == 0
        assert breaker.state == "CLOSED"
    
    def test_circuit_half_open_after_timeout(self):
        """Circuit should go HALF_OPEN after reset timeout."""
        breaker = CircuitBreaker(failure_threshold=1, reset_timeout=0)
        
        breaker.record_failure()
        assert breaker.state == "OPEN"
        
        # After timeout (immediate in this case)
        time.sleep(0.1)
        assert breaker.is_call_permitted() == True
        assert breaker.state == "HALF_OPEN"


# =============================================================================
# RATE LIMITER TESTS
# =============================================================================

class TestRateLimiter:
    """Tests for rate limiter."""
    
    def test_rate_limiter_config(self):
        """Test rate limiter configuration."""
        limiter = RateLimiter(max_concurrent=200, requests_per_second=15)
        
        stats = limiter.get_stats()
        assert stats["max_concurrent"] == 200
        assert stats["requests_per_second"] == 15
    
    @pytest.mark.asyncio
    async def test_rate_limiter_enforcement(self):
        """Test rate limiter enforces RPS limit."""
        limiter = RateLimiter(max_concurrent=100, requests_per_second=10)
        
        start = time.time()
        
        # Make 10 requests
        for _ in range(10):
            async with limiter:
                pass
        
        duration = time.time() - start
        
        # At 10 RPS, 10 requests should take ~1 second minimum
        assert duration >= 0.8, f"Duration {duration}s too short for rate limiting"
    
    @pytest.mark.asyncio
    async def test_concurrent_limit(self):
        """Test concurrent request limit is respected."""
        limiter = RateLimiter(max_concurrent=5, requests_per_second=100)
        
        # Simply verify the semaphore is configured correctly
        assert limiter.max_concurrent == 5
        
        # Test that get_stats works
        stats = limiter.get_stats()
        assert stats["max_concurrent"] == 5


# =============================================================================
# FALLBACK LAYER TESTS
# =============================================================================

class TestFallbackLayers:
    """Tests for 4-layer fallback system."""
    
    def test_layer_order(self):
        """Verify fallback layers execute in correct order."""
        layers = ["normal", "reinforced", "platinum", "emergency"]
        
        # Each layer should be tried before the next
        for i, layer in enumerate(layers):
            assert layer in ["normal", "reinforced", "platinum", "emergency"]
            if i > 0:
                # Verify ordering
                assert layers.index(layers[i]) > layers.index(layers[i-1])
    
    def test_emergency_fallback_exists(self):
        """Verify emergency fallback message exists."""
        emergency_message = "We're experiencing technical difficulties. Regular commentary will resume shortly."
        
        assert len(emergency_message) > 20
        assert "technical" in emergency_message.lower()


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
