"""
Template Performance Analytics

Tracks which debate templates get used most and their conversion rates.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class TemplateStats:
    """Statistics for a single template"""
    template_id: str
    name: str
    category: str = "general"
    
    # Usage
    times_selected: int = 0
    times_completed: int = 0
    times_exported: int = 0
    
    # Quality
    regenerations: int = 0
    avg_duration_seconds: float = 0
    total_duration_seconds: float = 0
    
    # Engagement
    avg_segments: float = 0
    total_segments: int = 0
    
    @property
    def completion_rate(self) -> float:
        if self.times_selected == 0:
            return 0
        return self.times_completed / self.times_selected * 100
    
    @property
    def export_rate(self) -> float:
        if self.times_completed == 0:
            return 0
        return self.times_exported / self.times_completed * 100
    
    @property
    def regeneration_rate(self) -> float:
        if self.total_segments == 0:
            return 0
        return self.regenerations / self.total_segments * 100


class TemplateAnalytics:
    """
    Track template usage and performance.
    
    Usage:
        analytics = TemplateAnalytics()
        
        # When template selected
        analytics.record_selection("head_to_head", "Head to Head")
        
        # When debate completed
        analytics.record_completion("head_to_head", duration=120, segments=6)
        
        # Get insights
        top = analytics.get_top_templates()
    """
    
    def __init__(self):
        self._templates: Dict[str, TemplateStats] = {}
        self._usage_log: List[dict] = []
    
    def _get_or_create(self, template_id: str, name: str = "", category: str = "general") -> TemplateStats:
        if template_id not in self._templates:
            self._templates[template_id] = TemplateStats(
                template_id=template_id,
                name=name or template_id,
                category=category
            )
        return self._templates[template_id]
    
    def record_selection(
        self, 
        template_id: str, 
        name: str = "",
        category: str = "general",
        creator_id: str = ""
    ):
        """Record template selection"""
        template = self._get_or_create(template_id, name, category)
        template.times_selected += 1
        
        self._usage_log.append({
            "event": "selection",
            "template_id": template_id,
            "creator_id": creator_id,
            "timestamp": time.time()
        })
    
    def record_completion(
        self, 
        template_id: str,
        duration_seconds: int = 0,
        segments: int = 0,
        creator_id: str = ""
    ):
        """Record template completion"""
        template = self._get_or_create(template_id)
        template.times_completed += 1
        template.total_duration_seconds += duration_seconds
        template.total_segments += segments
        
        # Update averages
        template.avg_duration_seconds = (
            template.total_duration_seconds / template.times_completed
        )
        template.avg_segments = (
            template.total_segments / template.times_completed
        )
        
        self._usage_log.append({
            "event": "completion",
            "template_id": template_id,
            "creator_id": creator_id,
            "duration": duration_seconds,
            "segments": segments,
            "timestamp": time.time()
        })
    
    def record_export(self, template_id: str, format_type: str = "", creator_id: str = ""):
        """Record template export"""
        template = self._get_or_create(template_id)
        template.times_exported += 1
        
        self._usage_log.append({
            "event": "export",
            "template_id": template_id,
            "format": format_type,
            "creator_id": creator_id,
            "timestamp": time.time()
        })
    
    def record_regeneration(self, template_id: str):
        """Record segment regeneration"""
        template = self._get_or_create(template_id)
        template.regenerations += 1
    
    def get_template_stats(self, template_id: str) -> Optional[dict]:
        """Get stats for a specific template"""
        if template_id not in self._templates:
            return None
        
        t = self._templates[template_id]
        return {
            "template_id": t.template_id,
            "name": t.name,
            "category": t.category,
            "usage": {
                "times_selected": t.times_selected,
                "times_completed": t.times_completed,
                "times_exported": t.times_exported,
            },
            "rates": {
                "completion_rate": round(t.completion_rate, 1),
                "export_rate": round(t.export_rate, 1),
                "regeneration_rate": round(t.regeneration_rate, 1),
            },
            "averages": {
                "duration_seconds": round(t.avg_duration_seconds, 1),
                "segments": round(t.avg_segments, 1),
            }
        }
    
    def get_top_templates(self, by: str = "usage", limit: int = 10) -> List[dict]:
        """Get top templates by a metric"""
        if by == "usage":
            key = lambda t: t.times_selected
        elif by == "completion":
            key = lambda t: t.completion_rate
        elif by == "export":
            key = lambda t: t.export_rate
        else:
            key = lambda t: t.times_selected
        
        sorted_templates = sorted(self._templates.values(), key=key, reverse=True)[:limit]
        
        return [
            {
                "rank": i + 1,
                "template_id": t.template_id,
                "name": t.name,
                "times_selected": t.times_selected,
                "completion_rate": round(t.completion_rate, 1),
                "export_rate": round(t.export_rate, 1),
            }
            for i, t in enumerate(sorted_templates)
        ]
    
    def get_summary(self) -> dict:
        """Get overall template analytics"""
        templates = list(self._templates.values())
        
        if not templates:
            return {"total_templates": 0}
        
        total_selections = sum(t.times_selected for t in templates)
        total_completions = sum(t.times_completed for t in templates)
        total_exports = sum(t.times_exported for t in templates)
        
        return {
            "total_templates": len(templates),
            "total_selections": total_selections,
            "total_completions": total_completions,
            "total_exports": total_exports,
            "overall_completion_rate": (
                round(total_completions / total_selections * 100, 1)
                if total_selections > 0 else 0
            ),
            "overall_export_rate": (
                round(total_exports / total_completions * 100, 1)
                if total_completions > 0 else 0
            ),
            "most_popular": self.get_top_templates(limit=5),
        }
    
    def get_category_breakdown(self) -> Dict[str, dict]:
        """Get stats by category"""
        categories: Dict[str, dict] = defaultdict(lambda: {
            "templates": 0, "selections": 0, "completions": 0
        })
        
        for t in self._templates.values():
            cat = categories[t.category]
            cat["templates"] += 1
            cat["selections"] += t.times_selected
            cat["completions"] += t.times_completed
        
        return dict(categories)


# Singleton
_analytics: Optional[TemplateAnalytics] = None

def get_template_analytics() -> TemplateAnalytics:
    global _analytics
    if _analytics is None:
        _analytics = TemplateAnalytics()
    return _analytics
