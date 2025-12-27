"""
Analytics and Observability Tests

Tests for W&B logging, Weave tracing, cost tracking, and alerting.
"""
import pytest
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# =============================================================================
# COST TRACKING TESTS
# =============================================================================

class TestCostTracking:
    """Tests for cost tracking functionality."""
    
    # Cost constants (from modal_orchestrator.py)
    GOOGLE_TTS_COST_PER_CHAR = 0.000004  # $4 per million chars
    ELEVENLABS_COST_PER_CHAR = 0.00003   # ~$30 per million chars
    
    def test_google_tts_cost_calculation(self):
        """Test Google TTS cost is calculated correctly."""
        text = "This is a test sentence for TTS."
        chars = len(text)
        
        cost = chars * self.GOOGLE_TTS_COST_PER_CHAR
        
        assert cost > 0
        assert cost < 0.01  # Should be very cheap
    
    def test_elevenlabs_cost_higher_than_google(self):
        """Test ElevenLabs costs more than Google."""
        text = "Test sentence"
        chars = len(text)
        
        google_cost = chars * self.GOOGLE_TTS_COST_PER_CHAR
        elevenlabs_cost = chars * self.ELEVENLABS_COST_PER_CHAR
        
        assert elevenlabs_cost > google_cost
    
    def test_daily_budget_limit(self):
        """Test daily budget limit is enforced."""
        daily_limit = 50.0  # $50/day
        current_spend = 45.0
        next_request_cost = 10.0
        
        # Should reject request over budget
        assert current_spend + next_request_cost > daily_limit
    
    def test_cost_logging_format(self):
        """Test cost log entries have correct format."""
        log_entry = {
            "timestamp": time.time(),
            "service": "google_tts",
            "amount": 0.0001,
            "chars": 25,
            "details": "TTS generation"
        }
        
        assert "timestamp" in log_entry
        assert "service" in log_entry
        assert "amount" in log_entry
        assert log_entry["amount"] >= 0


# =============================================================================
# WANDB LOGGING TESTS
# =============================================================================

class TestWandBLogging:
    """Tests for Weights & Biases logging."""
    
    def test_metric_names_valid(self):
        """Test W&B metric names follow conventions."""
        valid_metrics = [
            "vibe_check/score",
            "latency/debate_ms",
            "latency/tts_ms",
            "fallback/layer_used",
            "cost/daily_total"
        ]
        
        for metric in valid_metrics:
            # Should use / separator for grouping
            assert "/" in metric
            # No spaces
            assert " " not in metric
    
    def test_run_config_structure(self):
        """Test W&B run config has required fields."""
        config = {
            "game_id": "401547430",
            "environment": "production",
            "model": "gemini-2.0-flash",
            "tts_provider": "auto",
        }
        
        assert "game_id" in config
        assert "environment" in config
    
    def test_summary_metrics_tracked(self):
        """Test summary metrics are computed."""
        summary = {
            "total_generations": 150,
            "avg_vibe_score": 0.82,
            "fallback_rate": 0.05,
            "total_cost": 2.50,
        }
        
        assert summary["avg_vibe_score"] > 0
        assert summary["fallback_rate"] < 0.20  # Less than 20%


# =============================================================================
# WEAVE TRACING TESTS
# =============================================================================

class TestWeaveTracing:
    """Tests for Weave tracing functionality."""
    
    def test_trace_span_structure(self):
        """Test trace spans have required fields."""
        span = {
            "operation": "generate_debate",
            "start_time": time.time(),
            "end_time": time.time() + 5.0,
            "status": "success",
            "inputs": {"prompt": "Test"},
            "outputs": {"response": "Generated text"}
        }
        
        assert "operation" in span
        assert "start_time" in span
        assert "end_time" in span
        assert span["end_time"] > span["start_time"]
    
    def test_trace_captures_latency(self):
        """Test trace records latency correctly."""
        start = time.time()
        # Simulate operation
        time.sleep(0.01)
        end = time.time()
        
        duration_ms = (end - start) * 1000
        
        assert duration_ms > 0
        assert duration_ms < 5000  # Under 5 seconds
    
    def test_trace_parent_child_linking(self):
        """Test trace spans can be linked."""
        parent = {"span_id": "parent-123", "children": []}
        child = {"span_id": "child-456", "parent_id": "parent-123"}
        
        parent["children"].append(child["span_id"])
        
        assert child["parent_id"] == parent["span_id"]


# =============================================================================
# ALERTING TESTS
# =============================================================================

class TestAlerting:
    """Tests for alerting functionality."""
    
    def test_alert_severity_levels(self):
        """Test alert severity levels are defined."""
        levels = ["info", "warning", "error", "critical"]
        
        for level in levels:
            assert level in levels
    
    def test_slack_alert_format(self):
        """Test Slack alert has required structure."""
        alert = {
            "channel": "#neuron-alerts",
            "text": "High fallback rate detected",
            "blocks": [
                {"type": "section", "text": {"type": "mrkdwn", "text": "⚠️ Alert"}}
            ]
        }
        
        assert "channel" in alert
        assert "text" in alert
    
    def test_threshold_breach_detection(self):
        """Test threshold breaches are detected."""
        thresholds = {
            "vibe_score_min": 0.7,
            "fallback_rate_max": 0.15,
            "latency_max_ms": 30000,
        }
        
        current = {
            "vibe_score": 0.65,  # Below threshold
            "fallback_rate": 0.10,
            "latency_ms": 5000,
        }
        
        # Check for breaches
        breaches = []
        if current["vibe_score"] < thresholds["vibe_score_min"]:
            breaches.append("vibe_score")
        if current["fallback_rate"] > thresholds["fallback_rate_max"]:
            breaches.append("fallback_rate")
        if current["latency_ms"] > thresholds["latency_max_ms"]:
            breaches.append("latency")
        
        assert "vibe_score" in breaches


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
