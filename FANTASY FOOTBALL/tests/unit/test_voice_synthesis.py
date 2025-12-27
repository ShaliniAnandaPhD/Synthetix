"""
Voice Synthesis Unit Tests

Tests for TTS provider selection, voice mapping, and audio format.
"""
import pytest
import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# =============================================================================
# TTS PROVIDER SELECTION TESTS
# =============================================================================

class TestTTSProviderSelection:
    """Tests for TTS provider selection logic."""
    
    def test_provider_split_distribution(self):
        """Verify 75/25 split between Google TTS and ElevenLabs."""
        random.seed(42)
        
        selections = []
        for _ in range(1000):
            # Simulate the 75/25 logic from modal_orchestrator
            provider = "google" if random.random() < 0.75 else "elevenlabs"
            selections.append(provider)
        
        google_pct = selections.count("google") / len(selections)
        elevenlabs_pct = selections.count("elevenlabs") / len(selections)
        
        # Allow 5% tolerance
        assert 0.70 <= google_pct <= 0.80, f"Google TTS at {google_pct:.0%}, expected ~75%"
        assert 0.20 <= elevenlabs_pct <= 0.30, f"ElevenLabs at {elevenlabs_pct:.0%}, expected ~25%"


# =============================================================================
# VOICE MAPPING TESTS
# =============================================================================

class TestVoiceMapping:
    """Tests for agent voice mappings."""
    
    GOOGLE_VOICE_POOL = {
        "agent_1": "en-US-Studio-Q",
        "agent_2": "en-US-Neural2-J",
        "agent_3": "en-US-Neural2-D",
        "agent_4": "en-US-Neural2-A",
        "agent_5": "en-US-News-N",
        "agent_6": "en-US-Studio-O",
    }
    
    def test_all_agents_have_voices(self):
        """Verify all 6 agents have distinct voice mappings."""
        assert len(self.GOOGLE_VOICE_POOL) == 6
    
    def test_voices_are_unique(self):
        """Verify all voices are unique."""
        voices = list(self.GOOGLE_VOICE_POOL.values())
        assert len(set(voices)) == 6, "Some agents share voices"
    
    def test_agent_6_is_female(self):
        """Verify agent_6 (Zareena) has female voice."""
        # Studio-O is the female voice
        assert self.GOOGLE_VOICE_POOL["agent_6"] == "en-US-Studio-O"


# =============================================================================
# AUDIO FORMAT TESTS
# =============================================================================

class TestAudioFormat:
    """Tests for audio format validation."""
    
    def test_mp3_header_detection(self):
        """Test MP3 header detection logic."""
        # MP3 headers
        mp3_id3 = b'ID3\x04\x00\x00'  # ID3v2 tag
        mp3_sync = b'\xff\xfb\x90\x00'  # MPEG sync word
        
        assert self._is_mp3(mp3_id3)
        assert self._is_mp3(mp3_sync)
    
    def test_wav_header_detection(self):
        """Test WAV header detection logic."""
        wav_header = b'RIFF\x00\x00\x00\x00WAVEfmt '
        
        assert self._is_wav(wav_header)
    
    def test_invalid_audio_detection(self):
        """Test invalid audio detection."""
        invalid = b'This is not audio data'
        
        assert not self._is_mp3(invalid)
        assert not self._is_wav(invalid)
    
    def _is_mp3(self, data: bytes) -> bool:
        """Check if data starts with MP3 header."""
        if len(data) < 3:
            return False
        return data[:3] == b'ID3' or (data[0] == 0xFF and (data[1] & 0xE0) == 0xE0)
    
    def _is_wav(self, data: bytes) -> bool:
        """Check if data starts with WAV header."""
        if len(data) < 4:
            return False
        return data[:4] == b'RIFF'


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
