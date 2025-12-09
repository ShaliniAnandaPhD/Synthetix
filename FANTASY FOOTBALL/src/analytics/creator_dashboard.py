"""
Creator Dashboard Analytics

Tracks per-creator usage, costs, and engagement metrics.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class CreatorMetrics:
    """Metrics for a single creator"""
    creator_id: str
    
    # Usage metrics
    debates_created: int = 0
    debates_completed: int = 0
    debates_exported: int = 0
    live_sessions: int = 0
    total_duration_seconds: int = 0
    
    # Cost metrics
    total_cost_usd: float = 0
    llm_cost_usd: float = 0
    tts_cost_usd: float = 0
    storage_cost_usd: float = 0
    
    # Engagement metrics
    avg_session_duration_seconds: int = 0
    last_active: Optional[float] = None
    first_seen: Optional[float] = None
    days_active: int = 0
    
    # Content metrics
    segments_generated: int = 0
    segments_regenerated: int = 0
    exports_by_format: Dict[str, int] = field(default_factory=dict)
    templates_used: Dict[str, int] = field(default_factory=dict)
    regions_used: Dict[str, int] = field(default_factory=dict)


@dataclass
class DailySnapshot:
    """Daily metrics snapshot"""
    date: str
    debates_created: int = 0
    cost_usd: float = 0
    active_creators: int = 0
    exports: int = 0


class CreatorDashboard:
    """
    Per-creator analytics dashboard.
    
    Features:
    - Track usage per creator
    - Cost attribution
    - Engagement analytics
    - Trend analysis
    
    Usage:
        dashboard = CreatorDashboard()
        
        # Track events
        dashboard.record_debate_created("creator123", cost=0.15)
        dashboard.record_export("creator123", "srt")
        
        # Get metrics
        metrics = dashboard.get_creator_metrics("creator123")
        summary = dashboard.get_summary()
    """
    
    def __init__(self, storage_client=None):
        self.storage = storage_client
        self._creators: Dict[str, CreatorMetrics] = {}
        self._daily_snapshots: Dict[str, DailySnapshot] = {}
        self._active_today: set = set()
    
    def _get_or_create(self, creator_id: str) -> CreatorMetrics:
        """Get or create creator metrics"""
        if creator_id not in self._creators:
            self._creators[creator_id] = CreatorMetrics(
                creator_id=creator_id,
                first_seen=time.time()
            )
        return self._creators[creator_id]
    
    def _today(self) -> str:
        """Get today's date string"""
        return datetime.now().strftime("%Y-%m-%d")
    
    def record_debate_created(
        self, 
        creator_id: str, 
        template: str = "default",
        regions: List[str] = None
    ):
        """Record a new debate created"""
        metrics = self._get_or_create(creator_id)
        metrics.debates_created += 1
        metrics.last_active = time.time()
        
        # Track template usage
        metrics.templates_used[template] = metrics.templates_used.get(template, 0) + 1
        
        # Track region usage
        if regions:
            for region in regions:
                metrics.regions_used[region] = metrics.regions_used.get(region, 0) + 1
        
        self._active_today.add(creator_id)
        self._update_daily_snapshot("debates_created", 1)
    
    def record_debate_completed(self, creator_id: str, duration_seconds: int = 0):
        """Record a debate completed"""
        metrics = self._get_or_create(creator_id)
        metrics.debates_completed += 1
        metrics.total_duration_seconds += duration_seconds
        metrics.last_active = time.time()
        
        # Update average
        if metrics.debates_completed > 0:
            metrics.avg_session_duration_seconds = (
                metrics.total_duration_seconds // metrics.debates_completed
            )
    
    def record_export(self, creator_id: str, format_type: str):
        """Record an export"""
        metrics = self._get_or_create(creator_id)
        metrics.debates_exported += 1
        metrics.exports_by_format[format_type] = (
            metrics.exports_by_format.get(format_type, 0) + 1
        )
        metrics.last_active = time.time()
        
        self._update_daily_snapshot("exports", 1)
    
    def record_segment_generated(self, creator_id: str, regenerated: bool = False):
        """Record a segment generation"""
        metrics = self._get_or_create(creator_id)
        metrics.segments_generated += 1
        if regenerated:
            metrics.segments_regenerated += 1
    
    def record_live_session(self, creator_id: str, duration_seconds: int = 0):
        """Record a live commentary session"""
        metrics = self._get_or_create(creator_id)
        metrics.live_sessions += 1
        metrics.total_duration_seconds += duration_seconds
        metrics.last_active = time.time()
    
    def record_cost(
        self, 
        creator_id: str, 
        amount_usd: float,
        category: str = "llm"
    ):
        """Record a cost"""
        metrics = self._get_or_create(creator_id)
        metrics.total_cost_usd += amount_usd
        
        if category == "llm":
            metrics.llm_cost_usd += amount_usd
        elif category == "tts":
            metrics.tts_cost_usd += amount_usd
        elif category == "storage":
            metrics.storage_cost_usd += amount_usd
        
        self._update_daily_snapshot("cost_usd", amount_usd)
    
    def get_creator_metrics(self, creator_id: str) -> Optional[dict]:
        """Get all metrics for a creator"""
        if creator_id not in self._creators:
            return None
        
        m = self._creators[creator_id]
        return {
            "creator_id": m.creator_id,
            "usage": {
                "debates_created": m.debates_created,
                "debates_completed": m.debates_completed,
                "debates_exported": m.debates_exported,
                "live_sessions": m.live_sessions,
                "segments_generated": m.segments_generated,
                "completion_rate": (
                    m.debates_completed / m.debates_created * 100
                    if m.debates_created > 0 else 0
                ),
            },
            "costs": {
                "total_usd": round(m.total_cost_usd, 4),
                "llm_usd": round(m.llm_cost_usd, 4),
                "tts_usd": round(m.tts_cost_usd, 4),
                "storage_usd": round(m.storage_cost_usd, 4),
                "avg_per_debate": (
                    round(m.total_cost_usd / m.debates_created, 4)
                    if m.debates_created > 0 else 0
                ),
            },
            "engagement": {
                "avg_session_seconds": m.avg_session_duration_seconds,
                "last_active": m.last_active,
                "first_seen": m.first_seen,
                "days_since_first": (
                    int((time.time() - m.first_seen) / 86400)
                    if m.first_seen else 0
                ),
            },
            "preferences": {
                "top_templates": dict(
                    sorted(m.templates_used.items(), key=lambda x: -x[1])[:5]
                ),
                "top_regions": dict(
                    sorted(m.regions_used.items(), key=lambda x: -x[1])[:5]
                ),
                "export_formats": m.exports_by_format,
            }
        }
    
    def get_summary(self) -> dict:
        """Get overall summary across all creators"""
        total_creators = len(self._creators)
        total_debates = sum(m.debates_created for m in self._creators.values())
        total_cost = sum(m.total_cost_usd for m in self._creators.values())
        total_exports = sum(m.debates_exported for m in self._creators.values())
        
        # Active today
        active_today = len(self._active_today)
        
        # Top creators by usage
        top_by_debates = sorted(
            self._creators.values(),
            key=lambda m: m.debates_created,
            reverse=True
        )[:10]
        
        return {
            "totals": {
                "creators": total_creators,
                "debates": total_debates,
                "exports": total_exports,
                "cost_usd": round(total_cost, 2),
            },
            "today": {
                "active_creators": active_today,
                "snapshot": self._get_today_snapshot(),
            },
            "top_creators": [
                {
                    "creator_id": m.creator_id,
                    "debates": m.debates_created,
                    "cost_usd": round(m.total_cost_usd, 2)
                }
                for m in top_by_debates
            ],
            "averages": {
                "debates_per_creator": (
                    round(total_debates / total_creators, 1)
                    if total_creators > 0 else 0
                ),
                "cost_per_debate": (
                    round(total_cost / total_debates, 4)
                    if total_debates > 0 else 0
                ),
            }
        }
    
    def get_creator_leaderboard(self, metric: str = "debates", limit: int = 10) -> List[dict]:
        """Get top creators by a specific metric"""
        if metric == "debates":
            key = lambda m: m.debates_created
        elif metric == "cost":
            key = lambda m: m.total_cost_usd
        elif metric == "exports":
            key = lambda m: m.debates_exported
        else:
            key = lambda m: m.debates_created
        
        sorted_creators = sorted(self._creators.values(), key=key, reverse=True)[:limit]
        
        return [
            {
                "rank": i + 1,
                "creator_id": m.creator_id,
                "debates_created": m.debates_created,
                "total_cost_usd": round(m.total_cost_usd, 2),
                "exports": m.debates_exported
            }
            for i, m in enumerate(sorted_creators)
        ]
    
    def _update_daily_snapshot(self, field: str, value: Any):
        """Update today's snapshot"""
        today = self._today()
        if today not in self._daily_snapshots:
            self._daily_snapshots[today] = DailySnapshot(date=today)
        
        snapshot = self._daily_snapshots[today]
        if field == "debates_created":
            snapshot.debates_created += value
        elif field == "cost_usd":
            snapshot.cost_usd += value
        elif field == "exports":
            snapshot.exports += value
        
        snapshot.active_creators = len(self._active_today)
    
    def _get_today_snapshot(self) -> dict:
        """Get today's snapshot"""
        today = self._today()
        if today not in self._daily_snapshots:
            return {"date": today, "debates_created": 0, "cost_usd": 0, "exports": 0}
        
        s = self._daily_snapshots[today]
        return {
            "date": s.date,
            "debates_created": s.debates_created,
            "cost_usd": round(s.cost_usd, 2),
            "exports": s.exports,
            "active_creators": s.active_creators
        }


# Singleton
_dashboard: Optional[CreatorDashboard] = None

def get_creator_dashboard() -> CreatorDashboard:
    global _dashboard
    if _dashboard is None:
        _dashboard = CreatorDashboard()
    return _dashboard
