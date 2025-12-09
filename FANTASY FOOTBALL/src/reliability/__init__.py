"""
Reliability Module

Circuit breakers, rate limiting, health checks, retry, and feature flags.
"""

from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitState,
    CircuitConfig,
    CircuitOpenError,
    get_circuit,
    get_circuit_registry
)

from .rate_limiter import (
    RateLimiter,
    RateLimitConfig,
    GameRateLimiter,
    get_rate_limiter,
    get_game_rate_limiter
)

from .health_check import (
    HealthChecker,
    HealthStatus,
    ServiceHealth,
    get_health_checker
)

from .retry import (
    retry,
    RetryExecutor,
    RetryConfig,
    RETRY_CONFIGS
)

from .feature_flags import (
    FeatureFlagManager,
    FeatureFlag,
    FlagType,
    get_feature_flags
)

__all__ = [
    # Circuit Breaker
    'CircuitBreaker', 'CircuitBreakerRegistry', 'CircuitState', 'CircuitConfig',
    'CircuitOpenError', 'get_circuit', 'get_circuit_registry',
    # Rate Limiter
    'RateLimiter', 'RateLimitConfig', 'GameRateLimiter',
    'get_rate_limiter', 'get_game_rate_limiter',
    # Health Check
    'HealthChecker', 'HealthStatus', 'ServiceHealth', 'get_health_checker',
    # Retry
    'retry', 'RetryExecutor', 'RetryConfig', 'RETRY_CONFIGS',
    # Feature Flags
    'FeatureFlagManager', 'FeatureFlag', 'FlagType', 'get_feature_flags',
]
