"""
Live Latency Dashboard

Real-time monitoring of E2E latency, cache performance, and system health.
Provides metrics for display in UI and logging to W&B.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics tracked"""
    E2E_LATENCY = "e2e_latency"
    AGENT_LATENCY = "agent_latency"
    VOICE_LATENCY = "voice_latency"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    FALLBACK = "fallback"
    ERROR = "error"


@dataclass
class LatencySample:
    """A single latency measurement"""
    metric_type: MetricType
    value_ms: float
    timestamp: float = field(default_factory=time.time)
    region: str = ""
    event_type: str = ""
    source: str = ""


@dataclass
class DashboardStats:
    """Current dashboard statistics"""
    # Latency metrics
    avg_e2e_latency_ms: float = 0
    p50_e2e_latency_ms: float = 0
    p95_e2e_latency_ms: float = 0
    p99_e2e_latency_ms: float = 0
    
    # Cache metrics
    cache_hit_rate: float = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    # Health metrics
    events_processed: int = 0
    commentary_generated: int = 0
    errors: int = 0
    fallbacks_used: int = 0
    
    # Current state
    active_sessions: int = 0
    queue_depth: int = 0
    uptime_seconds: int = 0


class LiveLatencyDashboard:
    """
    Real-time latency monitoring for live commentary.
    
    Features:
    - Rolling window latency percentiles
    - Cache hit/miss rates
    - Error tracking
    - W&B integration for logging
    
    Usage:
        dashboard = LiveLatencyDashboard()
        await dashboard.start()
        
        # Record metrics during operation
        dashboard.record_latency(MetricType.E2E_LATENCY, 125)
        dashboard.record_cache_hit()
        
        # Get current stats for UI
        stats = dashboard.get_stats()
    """
    
    # Rolling window size
    WINDOW_SIZE = 1000
    
    # Publish interval (seconds)
    PUBLISH_INTERVAL = 5
    
    def __init__(
        self,
        wandb_logger=None,
        on_stats_update: Optional[Callable[[DashboardStats], None]] = None,
        on_alert: Optional[Callable[[str, str], None]] = None,
    ):
        """
        Initialize dashboard.
        
        Args:
            wandb_logger: W&B run for logging
            on_stats_update: Callback when stats update
            on_alert: Callback for alerts (level, message)
        """
        self.wandb_logger = wandb_logger
        self.on_stats_update = on_stats_update
        self.on_alert = on_alert
        
        # Rolling windows for different metrics
        self._latency_window: deque = deque(maxlen=self.WINDOW_SIZE)
        self._samples: Dict[MetricType, deque] = {
            mt: deque(maxlen=self.WINDOW_SIZE) for mt in MetricType
        }
        
        # Counters
        self._counters = {
            "events": 0,
            "commentary": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
            "fallbacks": 0,
        }
        
        self._start_time = time.time()
        self._running = False
        self._publish_task: Optional[asyncio.Task] = None
        
        # Alert thresholds
        self._alert_thresholds = {
            "p95_latency_ms": 500,
            "cache_hit_rate_min": 0.5,
            "error_rate_max": 0.1,
        }
    
    async def start(self):
        """Start the dashboard"""
        self._running = True
        self._start_time = time.time()
        self._publish_task = asyncio.create_task(self._publish_loop())
        logger.info("Latency dashboard started")
    
    async def stop(self):
        """Stop the dashboard"""
        self._running = False
        if self._publish_task:
            self._publish_task.cancel()
            try:
                await self._publish_task
            except asyncio.CancelledError:
                pass
        logger.info("Latency dashboard stopped")
    
    def record_latency(
        self,
        metric_type: MetricType,
        value_ms: float,
        region: str = "",
        event_type: str = "",
        source: str = ""
    ):
        """Record a latency sample"""
        sample = LatencySample(
            metric_type=metric_type,
            value_ms=value_ms,
            region=region,
            event_type=event_type,
            source=source
        )
        
        self._samples[metric_type].append(sample)
        
        if metric_type == MetricType.E2E_LATENCY:
            self._latency_window.append(value_ms)
    
    def record_cache_hit(self):
        """Record a cache hit"""
        self._counters["cache_hits"] += 1
        self.record_latency(MetricType.CACHE_HIT, 0)
    
    def record_cache_miss(self):
        """Record a cache miss"""
        self._counters["cache_misses"] += 1
        self.record_latency(MetricType.CACHE_MISS, 0)
    
    def record_event(self):
        """Record an event processed"""
        self._counters["events"] += 1
    
    def record_commentary(self):
        """Record commentary generated"""
        self._counters["commentary"] += 1
    
    def record_error(self):
        """Record an error"""
        self._counters["errors"] += 1
        self.record_latency(MetricType.ERROR, 0)
    
    def record_fallback(self):
        """Record a fallback used"""
        self._counters["fallbacks"] += 1
        self.record_latency(MetricType.FALLBACK, 0)
    
    def get_stats(self, active_sessions: int = 0, queue_depth: int = 0) -> DashboardStats:
        """Get current dashboard statistics"""
        latencies = list(self._latency_window)
        
        if latencies:
            sorted_latencies = sorted(latencies)
            n = len(sorted_latencies)
            
            avg = sum(latencies) / n
            p50 = sorted_latencies[int(n * 0.50)]
            p95 = sorted_latencies[int(n * 0.95)]
            p99 = sorted_latencies[int(n * 0.99)] if n >= 100 else p95
        else:
            avg = p50 = p95 = p99 = 0
        
        total_cache = self._counters["cache_hits"] + self._counters["cache_misses"]
        hit_rate = (
            self._counters["cache_hits"] / total_cache
            if total_cache > 0 else 0
        )
        
        return DashboardStats(
            avg_e2e_latency_ms=round(avg, 1),
            p50_e2e_latency_ms=round(p50, 1),
            p95_e2e_latency_ms=round(p95, 1),
            p99_e2e_latency_ms=round(p99, 1),
            cache_hit_rate=round(hit_rate, 3),
            cache_hits=self._counters["cache_hits"],
            cache_misses=self._counters["cache_misses"],
            events_processed=self._counters["events"],
            commentary_generated=self._counters["commentary"],
            errors=self._counters["errors"],
            fallbacks_used=self._counters["fallbacks"],
            active_sessions=active_sessions,
            queue_depth=queue_depth,
            uptime_seconds=int(time.time() - self._start_time),
        )
    
    def get_recent_latencies(self, count: int = 50) -> List[float]:
        """Get recent latency samples for charting"""
        return list(self._latency_window)[-count:]
    
    def _check_alerts(self, stats: DashboardStats):
        """Check for alert conditions"""
        if not self.on_alert:
            return
        
        # High latency alert
        if stats.p95_e2e_latency_ms > self._alert_thresholds["p95_latency_ms"]:
            self.on_alert(
                "warning",
                f"P95 latency {stats.p95_e2e_latency_ms}ms exceeds threshold"
            )
        
        # Low cache hit rate
        if stats.cache_hit_rate < self._alert_thresholds["cache_hit_rate_min"]:
            self.on_alert(
                "warning",
                f"Cache hit rate {stats.cache_hit_rate:.1%} below threshold"
            )
        
        # High error rate
        if stats.events_processed > 0:
            error_rate = stats.errors / stats.events_processed
            if error_rate > self._alert_thresholds["error_rate_max"]:
                self.on_alert(
                    "error",
                    f"Error rate {error_rate:.1%} exceeds threshold"
                )
    
    async def _publish_loop(self):
        """Periodically publish stats"""
        while self._running:
            try:
                await asyncio.sleep(self.PUBLISH_INTERVAL)
                
                stats = self.get_stats()
                
                # Check alerts
                self._check_alerts(stats)
                
                # Notify listeners
                if self.on_stats_update:
                    self.on_stats_update(stats)
                
                # Log to W&B
                if self.wandb_logger:
                    self.wandb_logger.log({
                        "e2e_latency_avg": stats.avg_e2e_latency_ms,
                        "e2e_latency_p95": stats.p95_e2e_latency_ms,
                        "cache_hit_rate": stats.cache_hit_rate,
                        "events_processed": stats.events_processed,
                        "errors": stats.errors,
                    })
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Dashboard publish error: {e}")
    
    def reset(self):
        """Reset all counters"""
        self._latency_window.clear()
        for samples in self._samples.values():
            samples.clear()
        for key in self._counters:
            self._counters[key] = 0
        self._start_time = time.time()


# Singleton
_dashboard: Optional[LiveLatencyDashboard] = None

def get_dashboard() -> LiveLatencyDashboard:
    """Get or create the global dashboard"""
    global _dashboard
    if _dashboard is None:
        _dashboard = LiveLatencyDashboard()
    return _dashboard


if __name__ == "__main__":
    import random
    
    async def test():
        alerts = []
        
        dashboard = LiveLatencyDashboard(
            on_alert=lambda level, msg: alerts.append((level, msg))
        )
        
        await dashboard.start()
        
        # Simulate metrics
        for _ in range(100):
            dashboard.record_latency(
                MetricType.E2E_LATENCY,
                random.uniform(50, 400)
            )
            dashboard.record_event()
            
            if random.random() > 0.3:
                dashboard.record_cache_hit()
            else:
                dashboard.record_cache_miss()
        
        stats = dashboard.get_stats()
        
        print(f"\n📊 Dashboard Stats")
        print(f"  Avg Latency: {stats.avg_e2e_latency_ms}ms")
        print(f"  P50 Latency: {stats.p50_e2e_latency_ms}ms")
        print(f"  P95 Latency: {stats.p95_e2e_latency_ms}ms")
        print(f"  Cache Hit Rate: {stats.cache_hit_rate:.1%}")
        print(f"  Events: {stats.events_processed}")
        print(f"  Alerts: {alerts}")
        
        await dashboard.stop()
    
    asyncio.run(test())
