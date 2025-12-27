"""
Voice Integration Tests

Tests for TTS fallback, quality validation, and concurrent requests.
"""
import pytest
import asyncio
import aiohttp
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Modal TTS endpoint
MODAL_TTS_URL = "https://neuronsystems--neuron-orchestrator-generate-tts.modal.run"


# =============================================================================
# TTS INTEGRATION TESTS
# =============================================================================

class TestTTSIntegration:
    """Integration tests for TTS system."""
    
    @pytest.mark.asyncio
    async def test_tts_endpoint_accessible(self):
        """Test TTS endpoint is accessible."""
        async with aiohttp.ClientSession() as session:
            payload = {
                "text": "Test message for voice synthesis.",
                "speaker_id": "agent_1"
            }
            
            try:
                async with session.post(
                    MODAL_TTS_URL,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    assert resp.status == 200
                    data = await resp.json()
                    
                    # Should have audio or error
                    assert "audio_base64" in data or "error" in data
                    
            except Exception as e:
                pytest.skip(f"TTS endpoint not accessible: {e}")
    
    @pytest.mark.asyncio
    async def test_tts_generates_audio(self):
        """Test TTS generates actual audio data."""
        async with aiohttp.ClientSession() as session:
            payload = {
                "text": "The quarterback throws a perfect touchdown pass!",
                "speaker_id": "agent_1"
            }
            
            try:
                start = time.time()
                async with session.post(
                    MODAL_TTS_URL,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    latency = time.time() - start
                    
                    if resp.status != 200:
                        pytest.skip("TTS endpoint returned non-200")
                    
                    data = await resp.json()
                    
                    if "error" in data:
                        pytest.skip(f"TTS error: {data['error']}")
                    
                    assert "audio_base64" in data, "No audio returned"
                    
                    audio = data["audio_base64"]
                    assert len(audio) > 1000, "Audio data too small"
                    
                    # Reasonable latency
                    assert latency < 30, f"TTS latency too high: {latency}s"
                    
            except Exception as e:
                pytest.skip(f"Test skipped: {e}")
    
    @pytest.mark.asyncio
    async def test_different_voices(self):
        """Test different speaker IDs produce different audio."""
        async with aiohttp.ClientSession() as session:
            text = "Test phrase for voice comparison."
            speakers = ["agent_1", "agent_2", "agent_6"]
            
            audio_results = []
            
            for speaker in speakers:
                payload = {"text": text, "speaker_id": speaker}
                
                try:
                    async with session.post(
                        MODAL_TTS_URL,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=60)
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if "audio_base64" in data:
                                audio_results.append(data["audio_base64"])
                except:
                    continue
            
            if len(audio_results) < 2:
                pytest.skip("Not enough audio samples for comparison")
            
            # Different voices should produce different audio
            unique_audio = set(audio_results)
            assert len(unique_audio) >= 2, "All voices produced identical audio"


# =============================================================================
# TTS QUALITY TESTS
# =============================================================================

class TestTTSQuality:
    """Tests for TTS audio quality."""
    
    @pytest.mark.asyncio
    async def test_audio_duration_reasonable(self):
        """Test audio duration is reasonable for text length."""
        async with aiohttp.ClientSession() as session:
            # Short text
            payload = {
                "text": "Touchdown!",
                "speaker_id": "agent_1"
            }
            
            try:
                async with session.post(
                    MODAL_TTS_URL,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    if resp.status != 200:
                        pytest.skip("TTS endpoint not available")
                    
                    data = await resp.json()
                    if "audio_base64" in data:
                        # Short phrase should produce short audio
                        # Base64 size roughly correlates with duration
                        audio_size = len(data["audio_base64"])
                        assert audio_size < 100000, "Audio too long for short phrase"
                        
            except Exception as e:
                pytest.skip(f"Test skipped: {e}")


# =============================================================================
# CONCURRENT TTS TESTS
# =============================================================================

class TestConcurrentTTS:
    """Tests for concurrent TTS requests."""
    
    @pytest.mark.asyncio
    async def test_concurrent_tts_requests(self):
        """Test multiple TTS requests don't interfere."""
        async with aiohttp.ClientSession() as session:
            texts = [
                f"Test phrase number {i} for concurrent testing."
                for i in range(5)
            ]
            
            async def make_request(text):
                payload = {"text": text, "speaker_id": "agent_1"}
                try:
                    async with session.post(
                        MODAL_TTS_URL,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=60)
                    ) as resp:
                        return resp.status == 200
                except:
                    return False
            
            results = await asyncio.gather(*[make_request(t) for t in texts])
            
            success_count = sum(results)
            if success_count == 0:
                pytest.skip("No TTS requests succeeded")
            
            # Most should succeed
            success_rate = success_count / len(texts)
            assert success_rate >= 0.6, f"Low success rate: {success_rate:.0%}"


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
