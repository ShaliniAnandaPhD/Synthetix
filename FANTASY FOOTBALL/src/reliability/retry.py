"""
Automatic Retry with Exponential Backoff

Handles transient failures gracefully.
"""

import logging
import asyncio
import random
import time
from functools import wraps
from typing import Callable, Optional, Tuple, Type, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Retry configuration"""
    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)


def calculate_delay(
    attempt: int,
    base_delay: float,
    max_delay: float,
    exponential_base: float,
    jitter: bool
) -> float:
    """Calculate delay for a given attempt"""
    delay = base_delay * (exponential_base ** attempt)
    delay = min(delay, max_delay)
    
    if jitter:
        delay = delay * (0.5 + random.random())
    
    return delay


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable] = None
):
    """
    Decorator for automatic retry with exponential backoff.
    
    Usage:
        @retry(max_attempts=3, base_delay=1.0)
        async def call_api():
            return await make_request()
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:
                        delay = calculate_delay(
                            attempt, base_delay, max_delay, exponential_base, jitter
                        )
                        
                        logger.warning(
                            f"Retry {attempt + 1}/{max_attempts} for {func.__name__} "
                            f"after {delay:.2f}s: {e}"
                        )
                        
                        if on_retry:
                            on_retry(func.__name__, attempt + 1, e)
                        
                        await asyncio.sleep(delay)
            
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:
                        delay = calculate_delay(
                            attempt, base_delay, max_delay, exponential_base, jitter
                        )
                        
                        logger.warning(
                            f"Retry {attempt + 1}/{max_attempts} for {func.__name__} "
                            f"after {delay:.2f}s: {e}"
                        )
                        
                        if on_retry:
                            on_retry(func.__name__, attempt + 1, e)
                        
                        time.sleep(delay)
            
            raise last_exception
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


class RetryExecutor:
    """
    Retry executor for runtime retry configuration.
    
    Usage:
        executor = RetryExecutor(max_attempts=3)
        result = await executor.execute(api_call, arg1, arg2)
    """
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
        on_retry: Optional[Callable] = None,
        on_success: Optional[Callable] = None,
        on_failure: Optional[Callable] = None
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions
        self.on_retry = on_retry
        self.on_success = on_success
        self.on_failure = on_failure
        
        self._stats = {"attempts": 0, "successes": 0, "failures": 0, "retries": 0}
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry"""
        last_exception = None
        
        for attempt in range(self.max_attempts):
            self._stats["attempts"] += 1
            
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                self._stats["successes"] += 1
                
                if self.on_success:
                    self.on_success(func.__name__, attempt + 1)
                
                return result
                
            except self.retryable_exceptions as e:
                last_exception = e
                
                if attempt < self.max_attempts - 1:
                    self._stats["retries"] += 1
                    
                    delay = calculate_delay(
                        attempt,
                        self.base_delay,
                        self.max_delay,
                        self.exponential_base,
                        self.jitter
                    )
                    
                    logger.warning(
                        f"Retry {attempt + 1}/{self.max_attempts}: {e}"
                    )
                    
                    if self.on_retry:
                        self.on_retry(func.__name__, attempt + 1, e)
                    
                    await asyncio.sleep(delay)
        
        self._stats["failures"] += 1
        
        if self.on_failure:
            self.on_failure(func.__name__, last_exception)
        
        raise last_exception
    
    def get_stats(self) -> dict:
        return {**self._stats}


# Common retry configurations
RETRY_CONFIGS = {
    "api": RetryConfig(
        max_attempts=3,
        base_delay_seconds=1.0,
        max_delay_seconds=10.0
    ),
    "database": RetryConfig(
        max_attempts=5,
        base_delay_seconds=0.5,
        max_delay_seconds=30.0
    ),
    "external_service": RetryConfig(
        max_attempts=3,
        base_delay_seconds=2.0,
        max_delay_seconds=60.0
    ),
}
