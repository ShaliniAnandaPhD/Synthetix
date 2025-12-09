"""
Agent-Level Traces

W&B integration for full pipeline tracing.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
from enum import Enum

logger = logging.getLogger(__name__)


class TraceStage(Enum):
    EVENT_RECEIVED = "event_received"
    EVENT_CLASSIFIED = "event_classified"
    AGENT_DISPATCHED = "agent_dispatched"
    AGENT_RESPONSE = "agent_response"
    VOICE_STARTED = "voice_started"
    VOICE_COMPLETE = "voice_complete"
    CLIENT_DELIVERED = "client_delivered"


@dataclass
class TraceSpan:
    """A single span in a trace"""
    stage: TraceStage
    start_time: float
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return 0
        return (self.end_time - self.start_time) * 1000


@dataclass
class AgentTrace:
    """Complete trace for an event through the pipeline"""
    trace_id: str
    event_id: str
    event_type: str
    region: str
    spans: List[TraceSpan] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    completed: bool = False
    error: Optional[str] = None
    
    @property
    def total_duration_ms(self) -> float:
        if not self.spans:
            return 0
        start = min(s.start_time for s in self.spans)
        end = max(s.end_time or s.start_time for s in self.spans)
        return (end - start) * 1000
    
    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "region": self.region,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "completed": self.completed,
            "error": self.error,
            "spans": [
                {
                    "stage": s.stage.value,
                    "duration_ms": round(s.duration_ms, 2),
                    "metadata": s.metadata
                }
                for s in self.spans
            ]
        }


class AgentTracer:
    """
    Traces events through the full pipeline.
    
    Usage:
        tracer = AgentTracer()
        
        # Start trace
        trace = tracer.start_trace("evt123", "touchdown", "kansas_city")
        
        # Add spans
        with tracer.span(trace.trace_id, TraceStage.AGENT_DISPATCHED):
            result = await dispatch_agent()
        
        # Complete
        tracer.complete_trace(trace.trace_id)
    """
    
    def __init__(self, wandb_run=None):
        self.wandb_run = wandb_run
        self._traces: Dict[str, AgentTrace] = {}
        self._completed_traces: List[AgentTrace] = []
        self._max_completed = 1000  # Keep last 1000 traces
    
    def start_trace(
        self, 
        event_id: str, 
        event_type: str, 
        region: str
    ) -> AgentTrace:
        """Start a new trace"""
        trace_id = f"{event_id}_{int(time.time() * 1000)}"
        trace = AgentTrace(
            trace_id=trace_id,
            event_id=event_id,
            event_type=event_type,
            region=region
        )
        self._traces[trace_id] = trace
        
        # Add initial span
        self.add_span(trace_id, TraceStage.EVENT_RECEIVED)
        
        return trace
    
    def add_span(
        self, 
        trace_id: str, 
        stage: TraceStage,
        metadata: Dict[str, Any] = None
    ) -> Optional[TraceSpan]:
        """Add a span to a trace"""
        if trace_id not in self._traces:
            return None
        
        span = TraceSpan(
            stage=stage,
            start_time=time.time(),
            metadata=metadata or {}
        )
        self._traces[trace_id].spans.append(span)
        return span
    
    def end_span(self, trace_id: str, stage: TraceStage):
        """End a span"""
        if trace_id not in self._traces:
            return
        
        for span in reversed(self._traces[trace_id].spans):
            if span.stage == stage and span.end_time is None:
                span.end_time = time.time()
                break
    
    @contextmanager
    def span(self, trace_id: str, stage: TraceStage, metadata: Dict = None):
        """Context manager for spans"""
        span = self.add_span(trace_id, stage, metadata)
        try:
            yield span
        finally:
            if span:
                span.end_time = time.time()
    
    def complete_trace(self, trace_id: str, error: str = None):
        """Mark trace as complete"""
        if trace_id not in self._traces:
            return
        
        trace = self._traces.pop(trace_id)
        trace.completed = True
        trace.error = error
        
        # End any open spans
        for span in trace.spans:
            if span.end_time is None:
                span.end_time = time.time()
        
        # Store completed trace
        self._completed_traces.append(trace)
        if len(self._completed_traces) > self._max_completed:
            self._completed_traces = self._completed_traces[-self._max_completed:]
        
        # Log to W&B
        if self.wandb_run:
            self._log_to_wandb(trace)
    
    def get_trace(self, trace_id: str) -> Optional[dict]:
        """Get trace by ID"""
        if trace_id in self._traces:
            return self._traces[trace_id].to_dict()
        
        for trace in self._completed_traces:
            if trace.trace_id == trace_id:
                return trace.to_dict()
        
        return None
    
    def get_recent_traces(self, limit: int = 50) -> List[dict]:
        """Get recent completed traces"""
        return [t.to_dict() for t in self._completed_traces[-limit:]]
    
    def get_stats(self) -> dict:
        """Get tracing statistics"""
        if not self._completed_traces:
            return {"total_traces": 0}
        
        durations = [t.total_duration_ms for t in self._completed_traces]
        errors = [t for t in self._completed_traces if t.error]
        
        return {
            "active_traces": len(self._traces),
            "completed_traces": len(self._completed_traces),
            "error_count": len(errors),
            "error_rate": len(errors) / len(self._completed_traces) * 100,
            "avg_duration_ms": sum(durations) / len(durations),
            "p50_duration_ms": sorted(durations)[len(durations) // 2],
            "p95_duration_ms": sorted(durations)[int(len(durations) * 0.95)],
        }
    
    def _log_to_wandb(self, trace: AgentTrace):
        """Log trace to W&B"""
        try:
            self.wandb_run.log({
                "trace/duration_ms": trace.total_duration_ms,
                "trace/event_type": trace.event_type,
                "trace/region": trace.region,
                "trace/error": 1 if trace.error else 0,
                "trace/span_count": len(trace.spans),
            })
        except Exception as e:
            logger.warning(f"Failed to log to W&B: {e}")


# Singleton
_tracer: Optional[AgentTracer] = None

def get_agent_tracer() -> AgentTracer:
    global _tracer
    if _tracer is None:
        _tracer = AgentTracer()
    return _tracer
