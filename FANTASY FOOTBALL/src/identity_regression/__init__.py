"""
Identity Regression Prevention Module

Provides tools for maintaining consistent AI sportscaster personalities
under high-load conditions.
"""

from .vibe_check import VibeCheckScorer, load_archetype_config
from .traces import load_platinum_traces, sample_platinum_traces, validate_platinum_archive
from .reinforcement import reinforce_identity
from .fallback import PlatinumFallbackSystem
from .rate_limiter import RateLimiter
from .event_mapping import map_event_to_type, get_event_importance
from .monitoring import (
    check_system_health, 
    send_slack_alert, 
    send_alerts_if_needed,
    format_metrics_summary
)

__all__ = [
    # Vibe checking
    "VibeCheckScorer",
    "load_archetype_config",
    
    # Platinum traces
    "load_platinum_traces",
    "sample_platinum_traces",
    "validate_platinum_archive",
    
    # Reinforcement
    "reinforce_identity",
    
    # Fallback
    "PlatinumFallbackSystem",
    
    # Rate limiting
    "RateLimiter",
    
    # Event mapping
    "map_event_to_type",
    "get_event_importance",
    
    # Monitoring
    "check_system_health",
    "send_slack_alert",
    "send_alerts_if_needed",
    "format_metrics_summary",
]
