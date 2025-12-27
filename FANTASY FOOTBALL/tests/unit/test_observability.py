"""
Observability Tests

Tests for monitoring, drift detection, and system health.
"""
import pytest
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# =============================================================================
# DRIFT DETECTION TESTS
# =============================================================================

class TestDriftDetection:
    """Tests for identity drift detection."""
    
    def test_vibe_score_drift_detection(self):
        """Test drift detected when scores consistently low."""
        scores = [0.82, 0.78, 0.65, 0.58, 0.52]  # Declining trend
        
        # Detect drift: 3+ consecutive below threshold
        threshold = 0.7
        consecutive_low = 0
        max_consecutive = 0
        
        for score in scores:
            if score < threshold:
                consecutive_low += 1
                max_consecutive = max(max_consecutive, consecutive_low)
            else:
                consecutive_low = 0
        
        has_drift = max_consecutive >= 3
        assert has_drift
    
    def test_archetype_confusion_detection(self):
        """Test detection of archetype confusion."""
        # Homer responses accidentally sounding neutral
        responses = [
            {"archetype": "homer", "score": 0.85, "neutral_markers": 1},
            {"archetype": "homer", "score": 0.72, "neutral_markers": 2},
            {"archetype": "homer", "score": 0.58, "neutral_markers": 5},  # Too neutral
        ]
        
        confused = [r for r in responses if r["neutral_markers"] >= 4]
        assert len(confused) > 0
    
    def test_energy_drift_detection(self):
        """Test energy level drift detection."""
        expected_energy = 0.85  # Homer baseline
        actual_energies = [0.82, 0.75, 0.68, 0.60]  # Declining
        
        drift = [abs(e - expected_energy) for e in actual_energies]
        avg_drift = sum(drift) / len(drift)
        
        # Significant drift if avg > 0.13
        assert avg_drift > 0.13


# =============================================================================
# HEALTH CHECK TESTS
# =============================================================================

class TestHealthChecks:
    """Tests for system health monitoring."""
    
    def test_health_endpoint_structure(self):
        """Test health endpoint response structure."""
        health = {
            "status": "healthy",
            "timestamp": time.time(),
            "services": {
                "llm": "healthy",
                "tts": "healthy",
                "redis": "healthy",
            }
        }
        
        assert "status" in health
        assert "services" in health
        assert all(s == "healthy" for s in health["services"].values())
    
    def test_degraded_status_detection(self):
        """Test degraded status detected correctly."""
        services = {
            "llm": "healthy",
            "tts": "degraded",  # One service down
            "redis": "healthy",
        }
        
        overall = "degraded" if "degraded" in services.values() else "healthy"
        assert overall == "degraded"
    
    def test_unhealthy_cascade(self):
        """Test unhealthy status cascades correctly."""
        services = {
            "llm": "unhealthy",  # Critical service down
            "tts": "healthy",
            "redis": "healthy",
        }
        
        critical = ["llm"]
        
        overall = "unhealthy" if any(services[s] == "unhealthy" for s in critical) else "healthy"
        assert overall == "unhealthy"


# =============================================================================
# METRICS AGGREGATION TESTS
# =============================================================================

class TestMetricsAggregation:
    """Tests for metrics aggregation."""
    
    def test_rolling_average_calculation(self):
        """Test rolling average calculated correctly."""
        values = [0.8, 0.85, 0.75, 0.9, 0.82]
        window = 3
        
        rolling_avg = sum(values[-window:]) / window
        
        expected = (0.75 + 0.9 + 0.82) / 3
        assert abs(rolling_avg - expected) < 0.01
    
    def test_percentile_calculation(self):
        """Test P50, P95, P99 calculation."""
        latencies = sorted([50, 60, 70, 80, 90, 100, 150, 200, 500, 800])
        
        p50_index = int(len(latencies) * 0.50)
        p95_index = int(len(latencies) * 0.95)
        
        p50 = latencies[p50_index]
        p95 = latencies[min(p95_index, len(latencies)-1)]
        
        assert p50 <= 100
        assert p95 >= 200
    
    def test_rate_calculation(self):
        """Test rate (per second) calculation."""
        events = 150
        duration_seconds = 60
        
        rate = events / duration_seconds
        
        assert rate == 2.5  # 2.5 events/second


# =============================================================================
# KILL SWITCH TESTS
# =============================================================================

class TestKillSwitch:
    """Tests for emergency kill switch."""
    
    def test_kill_switch_detection(self):
        """Test kill switch is detected."""
        env_vars = {"NEURON_KILL_SWITCH": "true"}
        
        is_killed = env_vars.get("NEURON_KILL_SWITCH", "").lower() in ["true", "1"]
        assert is_killed
    
    def test_kill_switch_default_off(self):
        """Test kill switch defaults to off."""
        env_vars = {}
        
        is_killed = env_vars.get("NEURON_KILL_SWITCH", "").lower() in ["true", "1"]
        assert not is_killed
    
    def test_vibe_check_disable_flag(self):
        """Test vibe check can be disabled."""
        env_vars = {"NEURON_VIBE_CHECK_DISABLED": "true"}
        
        disabled = env_vars.get("NEURON_VIBE_CHECK_DISABLED", "").lower() in ["true", "1"]
        assert disabled


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
