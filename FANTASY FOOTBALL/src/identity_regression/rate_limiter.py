"""
Rate limiter for API calls.
"""
import asyncio
import time
from asyncio import Semaphore


class RateLimiter:
    """
    Rate limiter for API calls.
    
    Game 2 worst case:
    - 150 creators responding simultaneously = 150 concurrent
    - Peak: 300 requests in 2 min + 15% regen = 345 total = 2.9 RPS
    - Headroom needed for spikes
    
    Default: max_concurrent=200, rps=15
    """
    
    def __init__(self, max_concurrent: int = 200, requests_per_second: float = 15):
        self.max_concurrent = max_concurrent
        self.requests_per_second = requests_per_second
        self.semaphore = Semaphore(max_concurrent)
        self.min_interval = 1.0 / requests_per_second
        self.last_request = 0
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire rate limit slot."""
        async with self.semaphore:
            async with self._lock:
                now = time.time()
                wait_time = self.min_interval - (now - self.last_request)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                self.last_request = time.time()
    
    async def __aenter__(self):
        await self.acquire()
        return self
    
    async def __aexit__(self, *args):
        pass
    
    def get_stats(self) -> dict:
        """Get current rate limiter stats."""
        return {
            "max_concurrent": self.max_concurrent,
            "requests_per_second": self.requests_per_second,
            "current_queue": self.max_concurrent - self.semaphore._value
        }
