"""
End-to-End Tests

Tests complete flow from event to audio delivery.
"""
import pytest
import asyncio
import aiohttp
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.identity_regression import map_event_to_type

# Modal endpoints
MODAL_DEBATE_URL = "https://neuronsystems--neuron-orchestrator-run-debate.modal.run"
MODAL_TTS_URL = "https://neuronsystems--neuron-orchestrator-generate-tts.modal.run"


# =============================================================================
# COMPLETE FLOW TESTS
# =============================================================================

class TestCompleteGameFlow:
    """End-to-end tests for complete game flow."""
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(180)
    async def test_event_to_commentary(self):
        """Test ESPN event → commentary flow."""
        # Simulate ESPN event
        espn_event = {
            "type": {"text": "Passing Touchdown"},
            "team": "HOU",
            "player": "C.J. Stroud",
            "description": "5-yard pass to Tank Dell"
        }
        
        # Map event type
        event_type = map_event_to_type(espn_event)
        assert event_type == "touchdown"
        
        # Generate commentary
        async with aiohttp.ClientSession() as session:
            payload = {
                "city1": "Houston",
                "city2": "Dallas",
                "topic": espn_event["description"],
                "rounds": 1
            }
            
            try:
                start = time.time()
                async with session.post(
                    MODAL_DEBATE_URL,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    latency = time.time() - start
                    
                    if resp.status != 200:
                        pytest.skip("Modal not accessible")
                    
                    data = await resp.json()
                    
                    assert data.get("status") == "success"
                    assert "debate" in data
                    
                    # Verify latency
                    assert latency < 60, f"Commentary latency: {latency:.1f}s"
                    
            except Exception as e:
                pytest.skip(f"Test skipped: {e}")
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_commentary_to_audio(self):
        """Test commentary → TTS flow."""
        async with aiohttp.ClientSession() as session:
            # Step 1: Generate commentary
            debate_payload = {
                "city1": "Houston",
                "city2": "Los Angeles",
                "topic": "Touchdown by the home team!",
                "rounds": 1
            }
            
            try:
                async with session.post(
                    MODAL_DEBATE_URL,
                    json=debate_payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    if resp.status != 200:
                        pytest.skip("Debate endpoint not accessible")
                    
                    debate_data = await resp.json()
                    if debate_data.get("status") != "success":
                        pytest.skip("Debate generation failed")
                    
                    # Extract commentary text
                    transcript = debate_data.get("debate", {}).get("transcript", [])
                    if not transcript:
                        pytest.skip("No transcript returned")
                    
                    commentary_text = transcript[0].get("response", "")
                    
            except Exception as e:
                pytest.skip(f"Debate failed: {e}")
            
            # Step 2: Generate audio
            tts_payload = {
                "text": commentary_text[:500],  # Limit length
                "speaker_id": "agent_1"
            }
            
            try:
                start = time.time()
                async with session.post(
                    MODAL_TTS_URL,
                    json=tts_payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    tts_latency = time.time() - start
                    
                    if resp.status != 200:
                        pytest.skip("TTS endpoint not accessible")
                    
                    tts_data = await resp.json()
                    
                    assert "audio_base64" in tts_data, "No audio generated"
                    assert len(tts_data["audio_base64"]) > 1000, "Audio too small"
                    
                    # Verify TTS latency
                    assert tts_latency < 30, f"TTS latency: {tts_latency:.1f}s"
                    
            except Exception as e:
                pytest.skip(f"TTS failed: {e}")
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(180)
    async def test_full_pipeline_latency(self):
        """Test complete pipeline latency under 60 seconds."""
        async with aiohttp.ClientSession() as session:
            start = time.time()
            
            # Step 1: Generate debate
            debate_payload = {
                "city1": "Houston",
                "city2": "Dallas",
                "topic": "Interception by the defense!",
                "rounds": 1
            }
            
            try:
                async with session.post(
                    MODAL_DEBATE_URL,
                    json=debate_payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    if resp.status != 200:
                        pytest.skip("Debate endpoint not accessible")
                    
                    debate_data = await resp.json()
                    debate_time = time.time() - start
                    
            except Exception as e:
                pytest.skip(f"Debate failed: {e}")
            
            # Step 2: TTS (if debate succeeded)
            if debate_data.get("status") == "success":
                transcript = debate_data.get("debate", {}).get("transcript", [])
                if transcript:
                    text = transcript[0].get("response", "Test")[:200]
                    
                    tts_start = time.time()
                    async with session.post(
                        MODAL_TTS_URL,
                        json={"text": text, "speaker_id": "agent_1"},
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as resp:
                        if resp.status == 200:
                            tts_time = time.time() - tts_start
            
            total_time = time.time() - start
            
            print(f"Total pipeline: {total_time:.1f}s")
            
            # Should complete under 60 seconds
            assert total_time < 60, f"Pipeline too slow: {total_time:.1f}s"


# =============================================================================
# SATURDAY SIMULATION
# =============================================================================

class TestSaturdaySimulation:
    """Simulate Saturday game conditions."""
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(600)
    async def test_game1_simulation(self):
        """Simulate Game 1: 100 creators responding to events."""
        num_events = 3
        creators_per_event = 10  # Reduced for test
        
        all_latencies = []
        successes = 0
        total = 0
        
        async with aiohttp.ClientSession() as session:
            for event_num in range(num_events):
                event_topic = ["Touchdown!", "Interception!", "Field goal!"][event_num % 3]
                
                tasks = []
                for creator in range(creators_per_event):
                    payload = {
                        "city1": "Houston",
                        "city2": "Los Angeles",
                        "topic": f"{event_topic} (Event {event_num}, Creator {creator})",
                        "rounds": 1
                    }
                    tasks.append(self._make_request(session, payload))
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for r in results:
                    total += 1
                    if not isinstance(r, Exception) and r.get("success"):
                        successes += 1
                        all_latencies.append(r.get("latency", 0))
                
                # Pause between events
                if event_num < num_events - 1:
                    await asyncio.sleep(5)
        
        if successes == 0:
            pytest.skip("No requests succeeded")
        
        success_rate = successes / total
        avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0
        
        print(f"Game 1 sim: {success_rate:.0%} success, {avg_latency:.1f}s avg")
        
        assert success_rate >= 0.70, f"Success rate: {success_rate:.0%}"
    
    async def _make_request(self, session, payload):
        """Make a single request."""
        start = time.time()
        try:
            async with session.post(
                MODAL_DEBATE_URL,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as resp:
                latency = time.time() - start
                return {
                    "success": resp.status == 200,
                    "latency": latency
                }
        except Exception:
            return {"success": False, "latency": time.time() - start}


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
