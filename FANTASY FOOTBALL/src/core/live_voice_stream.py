"""
Live Voice Stream for Live Commentary

Real-time audio streaming via WebSocket.
Handles chunked audio delivery for sub-300ms latency.
"""

import asyncio
import json
import base64
import logging
from dataclasses import dataclass
from typing import Optional, Dict, AsyncGenerator, Any
import time

logger = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    """A chunk of audio data for streaming"""
    region: str
    agent_id: str
    audio_base64: str
    text: str
    emotion: str
    chunk_index: int
    is_final: bool
    duration_ms: int


class LiveVoiceStream:
    """
    Streams synthesized voice to clients in real-time.
    Uses chunked audio for sub-300ms latency.
    
    Usage:
        voice_stream = LiveVoiceStream(voice_engine)
        await voice_stream.start_session(creator_id, websocket)
        
        # In your event loop:
        async for segment in tempo_coordinator.get_commentary_stream():
            await voice_stream.stream_segment(creator_id, segment)
    """
    
    def __init__(
        self, 
        voice_engine=None,
        cache_manager=None,
        chunk_size_ms: int = 500
    ):
        """
        Initialize voice stream.
        
        Args:
            voice_engine: Voice synthesis engine (ElevenLabs/Google)
            cache_manager: Optional cache for frequently used phrases
            chunk_size_ms: Size of audio chunks for streaming
        """
        self.voice_engine = voice_engine
        self.cache = cache_manager
        self.chunk_size_ms = chunk_size_ms
        self.active_sessions: Dict[str, Any] = {}
        self._running = False
    
    async def start(self):
        """Start the voice stream service"""
        self._running = True
        logger.info("LiveVoiceStream started")
    
    async def stop(self):
        """Stop the voice stream service"""
        self._running = False
        # Close all active sessions
        for session_id in list(self.active_sessions.keys()):
            await self.end_session(session_id)
        logger.info("LiveVoiceStream stopped")
    
    async def start_session(
        self, 
        creator_id: str, 
        websocket: Any,
        regional_voices: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Start a streaming session for a creator.
        
        Args:
            creator_id: Unique identifier for the creator
            websocket: WebSocket connection to the client
            regional_voices: Dict of region -> voice_id mapping
        
        Returns:
            True if session started successfully
        """
        if creator_id in self.active_sessions:
            logger.warning(f"Session {creator_id} already exists, replacing")
            await self.end_session(creator_id)
        
        self.active_sessions[creator_id] = {
            "websocket": websocket,
            "regional_voices": regional_voices or {},
            "started_at": time.time(),
            "segments_sent": 0,
            "bytes_sent": 0
        }
        
        # Send session start message
        await self._send_message(websocket, {
            "type": "session_start",
            "creator_id": creator_id,
            "timestamp": time.time()
        })
        
        logger.info(f"Started session for {creator_id}")
        return True
    
    async def end_session(self, creator_id: str):
        """End a streaming session"""
        if creator_id not in self.active_sessions:
            return
        
        session = self.active_sessions.pop(creator_id)
        
        try:
            await self._send_message(session["websocket"], {
                "type": "session_end",
                "creator_id": creator_id,
                "stats": {
                    "segments_sent": session["segments_sent"],
                    "bytes_sent": session["bytes_sent"],
                    "duration_seconds": time.time() - session["started_at"]
                }
            })
        except Exception as e:
            logger.warning(f"Error sending session end: {e}")
        
        logger.info(f"Ended session for {creator_id}")
    
    async def stream_segment(
        self, 
        creator_id: str, 
        segment: Any  # CommentarySegment
    ) -> bool:
        """
        Stream a commentary segment to the client.
        
        Args:
            creator_id: Session identifier
            segment: CommentarySegment to synthesize and stream
        
        Returns:
            True if segment was streamed successfully
        """
        if creator_id not in self.active_sessions:
            logger.warning(f"No active session for {creator_id}")
            return False
        
        session = self.active_sessions[creator_id]
        websocket = session["websocket"]
        
        try:
            # Get voice ID for this region
            voice_id = segment.voice_id or session["regional_voices"].get(
                segment.region, 
                "onwK4e9ZLuTAKqWW03F9"  # Default voice
            )
            
            # Check cache first
            cache_key = self._cache_key(segment.text, voice_id)
            cached_audio = await self._get_cached(cache_key) if self.cache else None
            
            if cached_audio:
                # Stream cached audio
                await self._stream_audio_chunk(
                    websocket, 
                    cached_audio, 
                    segment,
                    is_cached=True
                )
            else:
                # Generate and stream in chunks
                async for audio_chunk in self._synthesize_chunked(
                    segment.text, 
                    voice_id, 
                    segment.emotion
                ):
                    await self._stream_audio_chunk(
                        websocket, 
                        audio_chunk, 
                        segment
                    )
                    
                    # Cache for future use
                    if self.cache:
                        await self._set_cached(cache_key, audio_chunk)
            
            session["segments_sent"] += 1
            return True
            
        except Exception as e:
            logger.error(f"Error streaming segment: {e}")
            return False
    
    async def _synthesize_chunked(
        self, 
        text: str, 
        voice_id: str, 
        emotion: str
    ) -> AsyncGenerator[str, None]:
        """
        Synthesize audio and yield chunks.
        
        For real implementation, this would use ElevenLabs streaming API.
        For now, returns mock audio data.
        """
        if self.voice_engine:
            # Use real voice engine
            async for chunk in self.voice_engine.stream_synthesis(
                text=text,
                voice_id=voice_id,
                emotion=emotion
            ):
                yield chunk
        else:
            # Mock: yield placeholder audio chunks
            # In production, this would be actual audio data
            chunk_count = max(1, len(text) // 50)  # ~50 chars per chunk
            
            for i in range(chunk_count):
                # Simulate synthesis latency
                await asyncio.sleep(0.05)
                
                # Mock audio data (would be real base64 audio in production)
                mock_audio = base64.b64encode(
                    f"MOCK_AUDIO_CHUNK_{i}_{text[:20]}".encode()
                ).decode()
                
                yield mock_audio
    
    async def _stream_audio_chunk(
        self, 
        websocket: Any, 
        audio_data: str, 
        segment: Any,
        is_cached: bool = False
    ):
        """Send an audio chunk to the client"""
        message = {
            "type": "audio",
            "region": segment.region,
            "agent": segment.agent_id,
            "text": segment.text,
            "emotion": segment.emotion,
            "audio_base64": audio_data,
            "cached": is_cached,
            "timestamp": time.time()
        }
        
        await self._send_message(websocket, message)
        
        # Track bytes sent
        if segment.region in [s.get("region") for s in self.active_sessions.values()]:
            # Find the right session and update stats
            for session in self.active_sessions.values():
                session["bytes_sent"] += len(audio_data)
    
    async def _send_message(self, websocket: Any, message: dict):
        """Send a JSON message via WebSocket"""
        try:
            if hasattr(websocket, 'send'):
                await websocket.send(json.dumps(message))
            elif hasattr(websocket, 'send_json'):
                await websocket.send_json(message)
            else:
                # Mock for testing
                logger.debug(f"Would send: {message['type']}")
        except Exception as e:
            logger.error(f"WebSocket send error: {e}")
            raise
    
    def _cache_key(self, text: str, voice_id: str) -> str:
        """Generate cache key for audio"""
        import hashlib
        content = f"{voice_id}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def _get_cached(self, key: str) -> Optional[str]:
        """Get cached audio data"""
        if self.cache and hasattr(self.cache, 'get'):
            return await self.cache.get(key)
        return None
    
    async def _set_cached(self, key: str, audio_data: str):
        """Cache audio data"""
        if self.cache and hasattr(self.cache, 'set'):
            await self.cache.set(key, audio_data)
    
    def get_session_stats(self, creator_id: str) -> Optional[dict]:
        """Get stats for a session"""
        if creator_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[creator_id]
        return {
            "creator_id": creator_id,
            "segments_sent": session["segments_sent"],
            "bytes_sent": session["bytes_sent"],
            "duration_seconds": time.time() - session["started_at"],
            "active": True
        }
    
    def get_all_sessions(self) -> list[str]:
        """Get all active session IDs"""
        return list(self.active_sessions.keys())


# Factory function
def create_voice_stream(voice_engine=None, cache_manager=None) -> LiveVoiceStream:
    """Create a configured voice stream instance"""
    return LiveVoiceStream(
        voice_engine=voice_engine,
        cache_manager=cache_manager
    )


if __name__ == "__main__":
    # Test the voice stream
    from .live_tempo_coordinator import CommentarySegment
    
    async def test():
        stream = LiveVoiceStream()
        await stream.start()
        
        # Mock websocket
        class MockWebSocket:
            async def send(self, data):
                msg = json.loads(data)
                print(f"WS Send: {msg['type']} - {msg.get('text', '')[:30]}...")
        
        ws = MockWebSocket()
        await stream.start_session("test_creator", ws)
        
        segment = CommentarySegment(
            agent_id="agent1",
            region="dallas",
            text="TOUCHDOWN! What a play by the Cowboys!",
            emotion="excited",
            voice_id="JBFqnCBsd6RMkjVDRZzb"
        )
        
        await stream.stream_segment("test_creator", segment)
        
        print("\nSession stats:", stream.get_session_stats("test_creator"))
        
        await stream.end_session("test_creator")
        await stream.stop()
    
    asyncio.run(test())
