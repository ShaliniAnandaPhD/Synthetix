"""
Latency Percentiles Tracker

P50/P95/P99 latency tracking per pipeline stage.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import deque
import statistics

logger = logging.getLogger(__name__)


@dataclass 
class LatencyWindow:
    """Rolling window of latency samples"""
    samples: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def add(self, value_ms: float):
        self.samples.append(value_ms)
    
    def percentile(self, p: float) -> float:
        if not self.samples:
            return 0
        sorted_samples = sorted(self.samples)
        idx = int(len(sorted_samples) * p)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]
    
    @property
    def p50(self) -> float:
        return self.percentile(0.50)
    
    @property
    def p95(self) -> float:
        return self.percentile(0.95)
    
    @property
    def p99(self) -> float:
        return self.percentile(0.99)
    
    @property
    def avg(self) -> float:
        if not self.samples:
            return 0
        return statistics.mean(self.samples)
    
    @property
    def count(self) -> int:
        return len(self.samples)


class LatencyTracker:
    """
    Tracks latency percentiles across pipeline stages.
    
    Stages:
    - event_classification: Time to classify event
    - agent_dispatch: Time to dispatch to agent
    - agent_generation: Time for LLM response
    - voice_synthesis: Time for TTS
    - total_e2e: End-to-end latency
    
    Usage:
        tracker = LatencyTracker()
        
        # Record latencies
        tracker.record("agent_generation", 150)
        tracker.record("voice_synthesis", 200)
        tracker.record("total_e2e", 450)
        
        # Get percentiles
        stats = tracker.get_stats()
    """
    
    STAGES = [
        "event_classification",
        "agent_dispatch",
        "agent_generation",
        "voice_synthesis",
        "cache_lookup",
        "total_e2e"
    ]
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._windows: Dict[str, LatencyWindow] = {
            stage: LatencyWindow(samples=deque(maxlen=window_size))
            for stage in self.STAGES
        }
    
    def record(self, stage: str, latency_ms: float):
        """Record a latency sample"""
        if stage not in self._windows:
            self._windows[stage] = LatencyWindow(samples=deque(maxlen=self.window_size))
        self._windows[stage].add(latency_ms)
    
    def get_percentiles(self, stage: str) -> Optional[dict]:
        """Get percentiles for a stage"""
        if stage not in self._windows:
            return None
        
        window = self._windows[stage]
        return {
            "stage": stage,
            "count": window.count,
            "avg_ms": round(window.avg, 2),
            "p50_ms": round(window.p50, 2),
            "p95_ms": round(window.p95, 2),
            "p99_ms": round(window.p99, 2),
        }
    
    def get_all_percentiles(self) -> Dict[str, dict]:
        """Get percentiles for all stages"""
        return {
            stage: self.get_percentiles(stage)
            for stage in self._windows
        }
    
    def get_stats(self) -> dict:
        """Get summary statistics"""
        e2e = self._windows.get("total_e2e")
        
        return {
            "stages": self.get_all_percentiles(),
            "summary": {
                "e2e_p50_ms": round(e2e.p50, 2) if e2e else 0,
                "e2e_p95_ms": round(e2e.p95, 2) if e2e else 0,
                "e2e_p99_ms": round(e2e.p99, 2) if e2e else 0,
                "total_samples": sum(w.count for w in self._windows.values()),
            },
            "timestamp": time.time()
        }
    
    def get_dashboard_data(self) -> dict:
        """Get data formatted for dashboard display"""
        data = {"stages": []}
        
        for stage in self.STAGES:
            if stage in self._windows:
                window = self._windows[stage]
                data["stages"].append({
                    "name": stage.replace("_", " ").title(),
                    "p50": round(window.p50, 1),
                    "p95": round(window.p95, 1),
                    "p99": round(window.p99, 1),
                    "samples": window.count
                })
        
        return data
    
    def check_sla(self, sla_thresholds: Dict[str, float]) -> List[dict]:
        """Check if stages meet SLA thresholds"""
        violations = []
        
        for stage, threshold_ms in sla_thresholds.items():
            if stage in self._windows:
                p95 = self._windows[stage].p95
                if p95 > threshold_ms:
                    violations.append({
                        "stage": stage,
                        "threshold_ms": threshold_ms,
                        "actual_p95_ms": round(p95, 2),
                        "exceeded_by_ms": round(p95 - threshold_ms, 2)
                    })
        
        return violations
    
    def reset(self, stage: str = None):
        """Reset samples for a stage or all stages"""
        if stage:
            if stage in self._windows:
                self._windows[stage] = LatencyWindow(samples=deque(maxlen=self.window_size))
        else:
            for s in self._windows:
                self._windows[s] = LatencyWindow(samples=deque(maxlen=self.window_size))


# Default SLA thresholds
DEFAULT_SLAS = {
    "event_classification": 10,   # 10ms
    "agent_dispatch": 20,         # 20ms
    "agent_generation": 500,      # 500ms
    "voice_synthesis": 300,       # 300ms
    "cache_lookup": 50,           # 50ms
    "total_e2e": 750,             # 750ms
}


# Singleton
_tracker: Optional[LatencyTracker] = None

def get_latency_tracker() -> LatencyTracker:
    global _tracker
    if _tracker is None:
        _tracker = LatencyTracker()
    return _tracker
