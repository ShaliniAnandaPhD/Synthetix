"""
TTS (Text-to-Speech) module for voice synthesis.

Provides both batch and streaming TTS interfaces:
- streaming_tts: ElevenLabs WebSocket streaming for real-time audio
- google_tts: Google Cloud TTS for standard quality (defined in modal_orchestrator)
"""

from .streaming_tts import (
    StreamingTTSClient,
    TTSBuffer,
    get_voice_id,
    generate_streaming_audio_sync,
    CITY_VOICE_MAP,
)

__all__ = [
    "StreamingTTSClient",
    "TTSBuffer",
    "get_voice_id",
    "generate_streaming_audio_sync",
    "CITY_VOICE_MAP",
]
