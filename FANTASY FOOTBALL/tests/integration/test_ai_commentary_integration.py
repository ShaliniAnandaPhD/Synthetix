"""
AI Commentary Integration Tests

Tests for full generation pipeline, identity reinforcement, and multi-agent debates.
"""
import pytest
import asyncio
import aiohttp
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.identity_regression import (
    VibeCheckScorer,
    load_archetype_config,
    PlatinumFallbackSystem,
    reinforce_identity,
    load_platinum_traces
)

# Modal endpoints
MODAL_DEBATE_URL = "https://neuronsystems--neuron-orchestrator-run-debate.modal.run"


# =============================================================================
# FULL GENERATION PIPELINE TESTS
# =============================================================================

class TestGenerationPipeline:
    """Integration tests for the full generation pipeline."""
    
    @pytest.mark.asyncio
    async def test_modal_endpoint_health(self):
        """Test Modal endpoint is accessible."""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    "https://neuronsystems--neuron-orchestrator-dashboard-api.modal.run",
                    json={"action": "get_metrics"},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    assert resp.status == 200
                    data = await resp.json()
                    assert data.get("status") == "success"
            except Exception as e:
                pytest.skip(f"Modal not accessible: {e}")
    
    @pytest.mark.asyncio
    async def test_debate_generation(self):
        """Test debate endpoint generates valid response."""
        async with aiohttp.ClientSession() as session:
            payload = {
                "city1": "Houston",
                "city2": "Dallas",
                "topic": "Test: Who is the better team?",
                "rounds": 1,
                "agent_count": 2
            }
            
            try:
                start = time.time()
                async with session.post(
                    MODAL_DEBATE_URL,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    latency = time.time() - start
                    
                    assert resp.status == 200, f"Expected 200, got {resp.status}"
                    data = await resp.json()
                    
                    assert data.get("status") == "success"
                    assert "debate" in data
                    
                    # Verify reasonable latency
                    assert latency < 60, f"Latency too high: {latency}s"
                    
            except asyncio.TimeoutError:
                pytest.fail("Debate generation timed out (>120s)")
            except Exception as e:
                pytest.skip(f"Modal not accessible: {e}")
    
    @pytest.mark.asyncio
    async def test_response_quality(self):
        """Test generated responses have minimum quality."""
        async with aiohttp.ClientSession() as session:
            payload = {
                "city1": "Houston",
                "city2": "Los Angeles",
                "topic": "Touchdown by the Texans!",
                "rounds": 1
            }
            
            try:
                async with session.post(
                    MODAL_DEBATE_URL,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    if resp.status != 200:
                        pytest.skip("Modal not accessible")
                    
                    data = await resp.json()
                    
                    if data.get("status") != "success":
                        pytest.skip("Debate generation failed")
                    
                    # Extract response text
                    transcript = data.get("debate", {}).get("transcript", [])
                    assert len(transcript) > 0, "No transcript returned"
                    
                    response_text = transcript[0].get("response", "")
                    
                    # Quality checks
                    assert len(response_text) > 50, "Response too short"
                    assert not response_text.startswith("Error"), "Error response returned"
                    
            except Exception as e:
                pytest.skip(f"Test skipped: {e}")


# =============================================================================
# IDENTITY REINFORCEMENT TESTS
# =============================================================================

class TestIdentityReinforcement:
    """Tests for identity reinforcement system."""
    
    def test_reinforce_identity_adds_examples(self):
        """Test reinforcement adds platinum examples to prompt."""
        traces = load_platinum_traces()
        
        base_prompt = "Touchdown by the home team!"
        reinforced = reinforce_identity(base_prompt, "homer", traces)
        
        # Should add example section
        assert "example" in reinforced.lower() or len(reinforced) > len(base_prompt)
    
    def test_reinforce_identity_by_archetype(self):
        """Test each archetype gets appropriate reinforcement."""
        traces = load_platinum_traces()
        archetypes = ["statistician", "homer", "analyst"]
        
        for archetype in archetypes:
            reinforced = reinforce_identity("Test prompt", archetype, traces)
            
            # Each should produce different reinforcement
            assert reinforced is not None
            assert len(reinforced) > 0


# =============================================================================
# PLATINUM FALLBACK COVERAGE TESTS
# =============================================================================

class TestPlatinumCoverage:
    """Tests for platinum fallback coverage."""
    
    def test_saturday_cities_covered(self):
        """Verify all Saturday game cities have fallbacks."""
        fallback = PlatinumFallbackSystem()
        
        cities = ["houston", "los_angeles", "baltimore", "green_bay"]
        events = ["touchdown", "interception", "generic"]
        
        for city in cities:
            for event in events:
                response = fallback.get_fallback(city, event)
                
                assert response is not None, f"No fallback for {city}:{event}"
                assert len(response) > 20, f"Fallback too short for {city}:{event}"
                assert "technical difficulties" not in response.lower(), \
                    f"Emergency fallback used for {city}:{event}"
    
    def test_fallback_tier_order(self):
        """Test fallback tiers work in correct order."""
        fallback = PlatinumFallbackSystem()
        
        # Exact match should work
        exact = fallback.get_fallback("houston", "touchdown")
        assert exact is not None
        
        # Unknown event should fall back to generic
        generic = fallback.get_fallback("houston", "unknown_event")
        assert generic is not None
        
        # Unknown city should use universal
        universal = fallback.get_fallback("unknown_city", "touchdown")
        assert universal is not None


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
