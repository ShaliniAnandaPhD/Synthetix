"""
Evaluation Framework for Cultural Cognition System.

Tracks quantitative and qualitative metrics for P0 enhancements:
- Engagement metrics (debate duration, return rate, message depth)
- Coherence metrics (context retention, argument consistency)
- Performance metrics (latency, cache hit rate)
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class DebateMetrics:
    """Metrics for a single debate session."""
    debate_id: str
    user_id: str
    city_name: str
    start_time: str
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    message_count: int = 0
    user_return: bool = False  # Did user come back for another debate?
    abtest_group: str = "control"  # "control" or "treatment"
    
    # Coherence metrics
    context_retention_score: Optional[float] = None  # 0.0 to 1.0
    argument_consistency_score: Optional[float] = None  # 0.0 to 1.0
    metacognitive_moments: int = 0  # Count of self-aware comments
    
    # Performance metrics
    avg_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None
    cache_hit_rate: Optional[float] = None
    
    # Qualitative feedback
    authenticity_rating: Optional[int] = None  # 1-5 Likert
    compelling_rating: Optional[int] = None  # 1-5 Likert
    would_debate_again: Optional[bool] = None


class EvaluationTracker:
    """Tracks and analyzes cultural cognition system performance."""
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize evaluation tracker.
        
        Args:
            data_dir: Directory to store evaluation data
        """
        if data_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            data_dir = os.path.join(project_root, 'data', 'evaluation')
        
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.metrics_file = os.path.join(self.data_dir, 'debate_metrics.jsonl')
    
    def start_debate(
        self, 
        debate_id: str, 
        user_id: str, 
        city_name: str,
        abtest_group: str = "control"
    ) -> DebateMetrics:
        """
        Record the start of a debate session.
        
        Args:
            debate_id: Unique identifier for this debate
            user_id: User identifier
            city_name: City profile being used
            abtest_group: "control" or "treatment"
        
        Returns:
            DebateMetrics object
        """
        metrics = DebateMetrics(
            debate_id=debate_id,
            user_id=user_id,
            city_name=city_name,
            start_time=datetime.utcnow().isoformat(),
            abtest_group=abtest_group
        )
        
        return metrics
    
    def end_debate(self, metrics: DebateMetrics) -> None:
        """
        Record the end of a debate and calculate metrics.
        
        Args:
            metrics: DebateMetrics object from start_debate
        """
        metrics.end_time = datetime.utcnow().isoformat()
        
        # Calculate duration
        if metrics.start_time and metrics.end_time:
            start = datetime.fromisoformat(metrics.start_time)
            end = datetime.fromisoformat(metrics.end_time)
            metrics.duration_seconds = (end - start).total_seconds()
        
        # Write to JSONL file
        self._write_metrics(metrics)
    
    def _write_metrics(self, metrics: DebateMetrics) -> None:
        """Write metrics to JSONL file."""
        with open(self.metrics_file, 'a', encoding='utf-8') as f:
            json.dump(asdict(metrics), f)
            f.write('\n')
    
    def get_aggregated_metrics(
        self, 
        abtest_group: Optional[str] = None,
        city_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate aggregated metrics across all debates.
        
        Args:
            abtest_group: Optional filter by A/B test group
            city_name: Optional filter by city
        
        Returns:
            Dictionary of aggregated metrics
        """
        if not os.path.exists(self.metrics_file):
            return {
                "total_debates": 0,
                "error": "No metrics data found"
            }
        
        debates = []
        with open(self.metrics_file, 'r', encoding='utf-8') as f:
            for line in f:
                debate = json.loads(line)
                
                # Apply filters
                if abtest_group and debate.get('abtest_group') != abtest_group:
                    continue
                if city_name and debate.get('city_name') != city_name:
                    continue
                
                debates.append(debate)
        
        if not debates:
            return {"total_debates": 0}
        
        # Calculate aggregated metrics
        total_debates = len(debates)
        durations = [d.get('duration_seconds') for d in debates if d.get('duration_seconds')]
        message_counts = [d.get('message_count', 0) for d in debates]
        return_rates = [d.get('user_return', False) for d in debates]
        
        # Engagement metrics
        avg_duration = sum(durations) / len(durations) if durations else 0
        avg_message_count = sum(message_counts) / total_debates if message_counts else 0
        return_rate = sum(return_rates) / total_debates if return_rates else 0
        
        # Coherence metrics
        metacog_counts = [d.get('metacognitive_moments', 0) for d in debates]
        avg_metacog = sum(metacog_counts) / total_debates if metacog_counts else 0
        
        # Qualitative metrics
        auth_ratings = [d.get('authenticity_rating') for d in debates if d.get('authenticity_rating')]
        comp_ratings = [d.get('compelling_rating') for d in debates if d.get('compelling_rating')]
        
        avg_auth = sum(auth_ratings) / len(auth_ratings) if auth_ratings else None
        avg_comp = sum(comp_ratings) / len(comp_ratings) if comp_ratings else None
        
        return {
            "total_debates": total_debates,
            "engagement": {
                "avg_duration_seconds": round(avg_duration, 2),
                "avg_message_count": round(avg_message_count, 2),
                "return_rate_percent": round(return_rate * 100, 2)
            },
            "coherence": {
                "avg_metacognitive_moments": round(avg_metacog, 2)
            },
            "qualitative": {
                "avg_authenticity_rating": round(avg_auth, 2) if avg_auth else None,
                "avg_compelling_rating": round(avg_comp, 2) if avg_comp else None,
                "n_responses": len(auth_ratings)
            }
        }
    
    def compare_abtest_groups(self) -> Dict[str, Any]:
        """
        Compare control vs treatment groups.
        
        Returns:
            Dictionary with comparative metrics
        """
        control_metrics = self.get_aggregated_metrics(abtest_group="control")
        treatment_metrics = self.get_aggregated_metrics(abtest_group="treatment")
        
        if control_metrics["total_debates"] == 0 or treatment_metrics["total_debates"] == 0:
            return {
                "error": "Insufficient data for comparison",
                "control": control_metrics,
                "treatment": treatment_metrics
            }
        
        # Calculate percentage improvements
        improvements = {}
        
        if control_metrics["engagement"]["avg_duration_seconds"] > 0:
            duration_improvement = (
                (treatment_metrics["engagement"]["avg_duration_seconds"] - 
                 control_metrics["engagement"]["avg_duration_seconds"]) /
                control_metrics["engagement"]["avg_duration_seconds"] * 100
            )
            improvements["duration_percent_change"] = round(duration_improvement, 2)
        
        if control_metrics["engagement"]["return_rate_percent"] > 0:
            return_improvement = (
                treatment_metrics["engagement"]["return_rate_percent"] - 
                control_metrics["engagement"]["return_rate_percent"]
            )
            improvements["return_rate_point_change"] = round(return_improvement, 2)
        
        return {
            "control": control_metrics,
            "treatment": treatment_metrics,
            "improvements": improvements
        }
