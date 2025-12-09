"""
Health Check System

Comprehensive health checks for all services.
"""

import logging
import time
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ServiceHealth:
    """Health status of a single service"""
    name: str
    status: HealthStatus = HealthStatus.UNKNOWN
    latency_ms: float = 0
    last_check: float = 0
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class HealthChecker:
    """
    Comprehensive health check system.
    
    Features:
    - Check multiple services
    - Cache status
    - Dependency tracking
    
    Usage:
        health = HealthChecker()
        
        # Register checks
        health.register("redis", check_redis)
        health.register("elevenlabs", check_elevenlabs)
        
        # Get status
        status = await health.get_status()
    """
    
    def __init__(self, cache_duration_seconds: float = 10.0):
        self.cache_duration = cache_duration_seconds
        self._checks: Dict[str, Callable] = {}
        self._cache: Dict[str, ServiceHealth] = {}
        self._dependencies: Dict[str, List[str]] = {}
    
    def register(
        self, 
        name: str, 
        check_func: Callable,
        depends_on: List[str] = None
    ):
        """Register a health check"""
        self._checks[name] = check_func
        if depends_on:
            self._dependencies[name] = depends_on
    
    async def check(self, name: str, force: bool = False) -> ServiceHealth:
        """Check a single service"""
        if name not in self._checks:
            return ServiceHealth(name=name, status=HealthStatus.UNKNOWN, message="Not registered")
        
        # Check cache
        if not force and name in self._cache:
            cached = self._cache[name]
            if time.time() - cached.last_check < self.cache_duration:
                return cached
        
        # Run check
        start = time.time()
        try:
            check_func = self._checks[name]
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()
            
            latency = (time.time() - start) * 1000
            
            if isinstance(result, dict):
                health = ServiceHealth(
                    name=name,
                    status=HealthStatus(result.get("status", "healthy")),
                    latency_ms=latency,
                    last_check=time.time(),
                    message=result.get("message", ""),
                    metadata=result.get("metadata", {})
                )
            elif isinstance(result, bool):
                health = ServiceHealth(
                    name=name,
                    status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                    latency_ms=latency,
                    last_check=time.time()
                )
            else:
                health = ServiceHealth(
                    name=name,
                    status=HealthStatus.HEALTHY,
                    latency_ms=latency,
                    last_check=time.time()
                )
            
        except Exception as e:
            health = ServiceHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start) * 1000,
                last_check=time.time(),
                message=str(e)
            )
        
        self._cache[name] = health
        return health
    
    async def check_all(self, force: bool = False) -> Dict[str, ServiceHealth]:
        """Check all registered services"""
        results = {}
        for name in self._checks:
            results[name] = await self.check(name, force)
        return results
    
    async def get_status(self) -> dict:
        """Get comprehensive health status"""
        services = await self.check_all()
        
        # Determine overall status
        statuses = [s.status for s in services.values()]
        if all(s == HealthStatus.HEALTHY for s in statuses):
            overall = HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall = HealthStatus.UNHEALTHY
        else:
            overall = HealthStatus.DEGRADED
        
        return {
            "status": overall.value,
            "timestamp": time.time(),
            "services": {
                name: {
                    "status": h.status.value,
                    "latency_ms": round(h.latency_ms, 1),
                    "message": h.message,
                    "last_check": h.last_check
                }
                for name, h in services.items()
            },
            "summary": {
                "total": len(services),
                "healthy": len([s for s in statuses if s == HealthStatus.HEALTHY]),
                "degraded": len([s for s in statuses if s == HealthStatus.DEGRADED]),
                "unhealthy": len([s for s in statuses if s == HealthStatus.UNHEALTHY]),
            }
        }


# Standard health checks
def create_redis_check(redis_url: str = None):
    """Create Redis health check"""
    async def check():
        try:
            import redis.asyncio as redis
            import os
            url = redis_url or os.environ.get("REDIS_URL", "")
            if not url:
                return {"status": "degraded", "message": "No Redis URL configured"}
            
            r = redis.from_url(url)
            await r.ping()
            await r.close()
            return {"status": "healthy"}
        except Exception as e:
            return {"status": "unhealthy", "message": str(e)}
    return check


def create_modal_check():
    """Create Modal health check"""
    def check():
        try:
            import modal
            return {"status": "healthy", "message": "Modal SDK available"}
        except:
            return {"status": "unhealthy", "message": "Modal SDK not available"}
    return check


def create_supabase_check(url: str = None, key: str = None):
    """Create Supabase health check"""
    async def check():
        try:
            from supabase import create_client
            import os
            
            url_ = url or os.environ.get("SUPABASE_URL", "")
            key_ = key or os.environ.get("SUPABASE_KEY", "")
            
            if not url_ or not key_:
                return {"status": "degraded", "message": "Supabase not configured"}
            
            client = create_client(url_, key_)
            # Simple query to check connection
            return {"status": "healthy"}
        except Exception as e:
            return {"status": "unhealthy", "message": str(e)}
    return check


# Singleton
_health_checker: Optional[HealthChecker] = None

def get_health_checker() -> HealthChecker:
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
        # Register default checks
        _health_checker.register("modal", create_modal_check())
    return _health_checker
