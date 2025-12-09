"""
Streaming TTS Client for ElevenLabs WebSocket API.

This module provides real-time text-to-speech conversion using ElevenLabs'
WebSocket streaming API for low-latency audio generation.

Features:
- Sub-500ms first audio chunk latency
- Streaming text input (generates audio as text arrives)
- Configurable voices per city/personality
- Audio chunking for smooth playback
"""

import os
import json
import asyncio
import base64
from typing import AsyncGenerator, Optional, Dict, Any, List


# Voice IDs for different cities (ElevenLabs voice library)
CITY_VOICE_MAP = {
    "Philadelphia": "pNInz6obpgDQGcFmaJgB",  # Adam - energetic, passionate
    "Dallas": "ErXwobaYiN019PkySvjV",         # Antoni - confident, bold
    "New York": "VR6AewLTigWG4xSOukaG",       # Arnold - assertive, direct
    "Kansas City": "pNInz6obpgDQGcFmaJgB",    # Adam - enthusiastic
    "San Francisco": "IKne3meq5aSn9XLyUdCD", # Charlie - articulate
    "default": "21m00Tcm4TlvDq8ikWAM",        # Rachel - neutral, clear
}

def get_voice_id(city_name: str) -> str:
    """Get ElevenLabs voice ID for a city."""
    return CITY_VOICE_MAP.get(city_name, CITY_VOICE_MAP["default"])


class StreamingTTSClient:
    """
    ElevenLabs WebSocket streaming TTS client.
    
    Usage:
        client = StreamingTTSClient()
        async for audio_chunk in client.stream_audio("Hello world!", "Philadelphia"):
            # Process audio chunk (bytes)
            pass
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the streaming TTS client.
        
        Args:
            api_key: ElevenLabs API key. If not provided, reads from ELEVENLABS_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError("ElevenLabs API key not found. Set ELEVENLABS_API_KEY env var.")
        
        self.base_url = "wss://api.elevenlabs.io/v1/text-to-speech"
        
        # Default voice settings
        self.default_settings = {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.5,
            "use_speaker_boost": True
        }
    
    async def stream_audio(
        self,
        text: str,
        city_name: str = "default",
        model_id: str = "eleven_turbo_v2",
        output_format: str = "mp3_44100_128"
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream text to audio using ElevenLabs WebSocket API.
        
        Args:
            text: Text to convert to speech
            city_name: City name for voice selection
            model_id: ElevenLabs model ID (turbo for speed)
            output_format: Audio format (mp3_44100_128 for quality)
        
        Yields:
            Audio chunks as bytes
        """
        import websockets
        
        voice_id = get_voice_id(city_name)
        ws_url = f"{self.base_url}/{voice_id}/stream-input?model_id={model_id}&output_format={output_format}"
        
        async with websockets.connect(
            ws_url,
            extra_headers={"xi-api-key": self.api_key}
        ) as websocket:
            # Send initial configuration
            await websocket.send(json.dumps({
                "text": " ",  # Initial flush
                "voice_settings": self.default_settings,
                "xi_api_key": self.api_key,
                "try_trigger_generation": False
            }))
            
            # Send the actual text
            await websocket.send(json.dumps({
                "text": text,
                "try_trigger_generation": True
            }))
            
            # Send end of stream signal
            await websocket.send(json.dumps({
                "text": ""
            }))
            
            # Receive audio chunks
            async for message in websocket:
                data = json.loads(message)
                
                if data.get("audio"):
                    # Decode base64 audio chunk
                    audio_chunk = base64.b64decode(data["audio"])
                    yield audio_chunk
                
                if data.get("isFinal"):
                    break
    
    async def stream_text_chunks(
        self,
        text_chunks: AsyncGenerator[str, None],
        city_name: str = "default",
        model_id: str = "eleven_turbo_v2"
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream multiple text chunks to audio (for LLM streaming output).
        
        Args:
            text_chunks: Async generator of text chunks
            city_name: City name for voice selection
            model_id: ElevenLabs model ID
        
        Yields:
            Audio chunks as bytes
        """
        import websockets
        
        voice_id = get_voice_id(city_name)
        ws_url = f"{self.base_url}/{voice_id}/stream-input?model_id={model_id}"
        
        async with websockets.connect(
            ws_url,
            extra_headers={"xi-api-key": self.api_key}
        ) as websocket:
            # Send initial configuration
            await websocket.send(json.dumps({
                "text": " ",
                "voice_settings": self.default_settings,
                "xi_api_key": self.api_key,
                "try_trigger_generation": False
            }))
            
            # Task to send text chunks
            async def send_chunks():
                async for chunk in text_chunks:
                    if chunk.strip():
                        await websocket.send(json.dumps({
                            "text": chunk,
                            "try_trigger_generation": True
                        }))
                
                # Signal end of input
                await websocket.send(json.dumps({"text": ""}))
            
            # Start sending chunks in background
            send_task = asyncio.create_task(send_chunks())
            
            # Receive audio chunks
            try:
                async for message in websocket:
                    data = json.loads(message)
                    
                    if data.get("audio"):
                        audio_chunk = base64.b64decode(data["audio"])
                        yield audio_chunk
                    
                    if data.get("isFinal"):
                        break
            finally:
                send_task.cancel()


class TTSBuffer:
    """
    Audio buffer for smooth playback.
    
    Collects audio chunks and provides buffered output
    to prevent playback gaps.
    """
    
    def __init__(self, min_buffer_size: int = 4096):
        """
        Initialize the TTS buffer.
        
        Args:
            min_buffer_size: Minimum bytes before releasing audio
        """
        self.buffer = b""
        self.min_buffer_size = min_buffer_size
        self.is_complete = False
    
    def add_chunk(self, chunk: bytes) -> Optional[bytes]:
        """
        Add a chunk to the buffer.
        
        Args:
            chunk: Audio bytes to add
        
        Returns:
            Buffered audio if minimum size reached, else None
        """
        self.buffer += chunk
        
        if len(self.buffer) >= self.min_buffer_size:
            output = self.buffer
            self.buffer = b""
            return output
        
        return None
    
    def flush(self) -> Optional[bytes]:
        """Flush remaining buffer contents."""
        if self.buffer:
            output = self.buffer
            self.buffer = b""
            self.is_complete = True
            return output
        return None


# =============================================================================
# SYNCHRONOUS WRAPPER (for non-async contexts)
# =============================================================================

def generate_streaming_audio_sync(
    text: str,
    city_name: str = "default",
    api_key: Optional[str] = None
) -> bytes:
    """
    Synchronous wrapper for streaming TTS.
    
    Collects all audio chunks and returns complete audio.
    
    Args:
        text: Text to convert
        city_name: City for voice selection
        api_key: Optional ElevenLabs API key
    
    Returns:
        Complete audio as bytes
    """
    async def collect_audio():
        client = StreamingTTSClient(api_key)
        audio_chunks = []
        
        async for chunk in client.stream_audio(text, city_name):
            audio_chunks.append(chunk)
        
        return b"".join(audio_chunks)
    
    return asyncio.run(collect_audio())
