"""
Circuit Breaker Pattern

Auto-disables failing services with gradual recovery.
"""

import logging
import time
import asyncio
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, Any
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitConfig:
    """Configuration for a circuit breaker"""
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 3          # Successes to close
    timeout_seconds: float = 30.0       # Time before half-open
    half_open_max_calls: int = 3        # Max calls in half-open


@dataclass
class CircuitStats:
    """Statistics for a circuit"""
    total_calls: int = 0
    failures: int = 0
    successes: int = 0
    rejections: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None


class CircuitBreaker:
    """
    Circuit breaker for a single service.
    
    States:
    - CLOSED: Normal, passes all calls through
    - OPEN: Failing, rejects all calls immediately
    - HALF_OPEN: Testing, allows limited calls
    
    Usage:
        circuit = CircuitBreaker("elevenlabs")
        
        result = await circuit.call(make_api_request, args)
        
        if circuit.is_open:
            # Use fallback
            pass
    """
    
    def __init__(
        self, 
        name: str, 
        config: Optional[CircuitConfig] = None,
        on_state_change: Optional[Callable[[str, CircuitState], None]] = None
    ):
        self.name = name
        self.config = config or CircuitConfig()
        self.on_state_change = on_state_change
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._stats = CircuitStats()
    
    @property
    def state(self) -> CircuitState:
        # Check if we should transition from OPEN to HALF_OPEN
        if self._state == CircuitState.OPEN:
            if self._last_failure_time:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.config.timeout_seconds:
                    self._transition(CircuitState.HALF_OPEN)
        return self._state
    
    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN
    
    @property
    def is_closed(self) -> bool:
        return self.state == CircuitState.CLOSED
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker"""
        state = self.state
        self._stats.total_calls += 1
        
        # Reject if open
        if state == CircuitState.OPEN:
            self._stats.rejections += 1
            raise CircuitOpenError(f"Circuit {self.name} is open")
        
        # Limit calls in half-open
        if state == CircuitState.HALF_OPEN:
            if self._half_open_calls >= self.config.half_open_max_calls:
                self._stats.rejections += 1
                raise CircuitOpenError(f"Circuit {self.name} half-open limit reached")
            self._half_open_calls += 1
        
        # Execute the call
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure()
            raise
    
    def record_success(self):
        """Manually record a success"""
        self._on_success()
    
    def record_failure(self):
        """Manually record a failure"""
        self._on_failure()
    
    def reset(self):
        """Force reset to closed state"""
        self._transition(CircuitState.CLOSED)
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
    
    def get_stats(self) -> dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "stats": {
                "total_calls": self._stats.total_calls,
                "failures": self._stats.failures,
                "successes": self._stats.successes,
                "rejections": self._stats.rejections,
            }
        }
    
    def _on_success(self):
        self._stats.successes += 1
        self._stats.last_success_time = time.time()
        
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.config.success_threshold:
                self._transition(CircuitState.CLOSED)
        else:
            self._failure_count = 0
    
    def _on_failure(self):
        self._stats.failures += 1
        self._stats.last_failure_time = time.time()
        self._last_failure_time = time.time()
        
        if self._state == CircuitState.HALF_OPEN:
            self._transition(CircuitState.OPEN)
        else:
            self._failure_count += 1
            if self._failure_count >= self.config.failure_threshold:
                self._transition(CircuitState.OPEN)
    
    def _transition(self, new_state: CircuitState):
        if new_state != self._state:
            old_state = self._state
            self._state = new_state
            
            if new_state == CircuitState.HALF_OPEN:
                self._half_open_calls = 0
                self._success_count = 0
            elif new_state == CircuitState.CLOSED:
                self._failure_count = 0
                self._success_count = 0
            
            logger.info(f"Circuit {self.name}: {old_state.value} -> {new_state.value}")
            
            if self.on_state_change:
                self.on_state_change(self.name, new_state)


class CircuitOpenError(Exception):
    """Raised when circuit is open"""
    pass


class CircuitBreakerRegistry:
    """
    Manages multiple circuit breakers.
    
    Usage:
        registry = CircuitBreakerRegistry()
        
        # Get or create a circuit
        circuit = registry.get("elevenlabs")
        
        # Check all circuits
        status = registry.get_all_status()
    """
    
    def __init__(self, on_state_change: Optional[Callable] = None):
        self._circuits: Dict[str, CircuitBreaker] = {}
        self._on_state_change = on_state_change
    
    def get(self, name: str, config: Optional[CircuitConfig] = None) -> CircuitBreaker:
        """Get or create a circuit breaker"""
        if name not in self._circuits:
            self._circuits[name] = CircuitBreaker(
                name=name,
                config=config,
                on_state_change=self._on_state_change
            )
        return self._circuits[name]
    
    def get_all_status(self) -> Dict[str, dict]:
        """Get status of all circuits"""
        return {
            name: circuit.get_stats()
            for name, circuit in self._circuits.items()
        }
    
    def reset_all(self):
        """Reset all circuits"""
        for circuit in self._circuits.values():
            circuit.reset()
    
    def get_open_circuits(self) -> list:
        """Get list of open circuits"""
        return [
            name for name, circuit in self._circuits.items()
            if circuit.is_open
        ]


# Global registry
_registry: Optional[CircuitBreakerRegistry] = None

def get_circuit_registry() -> CircuitBreakerRegistry:
    global _registry
    if _registry is None:
        _registry = CircuitBreakerRegistry()
    return _registry

def get_circuit(name: str) -> CircuitBreaker:
    return get_circuit_registry().get(name)
