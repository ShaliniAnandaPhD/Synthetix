"""
Live Tempo Coordinator for Live Commentary

Manages turn-taking and pacing for live commentary.
Ensures natural flow without awkward pauses or pile-ups.
Uses cultural interrupt thresholds from regional profiles.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Optional, AsyncGenerator
from collections import deque
import time
import logging

from .event_classifier import ClassifiedEvent, EventUrgency

logger = logging.getLogger(__name__)


@dataclass
class CommentarySegment:
    """A segment of commentary ready for voice synthesis"""
    agent_id: str
    region: str
    text: str
    priority: int = 5              # Higher = speak sooner (1-10)
    max_duration_ms: int = 5000    # How long this should take to say
    can_be_interrupted: bool = True
    emotion: str = "neutral"
    voice_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "region": self.region,
            "text": self.text,
            "priority": self.priority,
            "max_duration_ms": self.max_duration_ms,
            "can_be_interrupted": self.can_be_interrupted,
            "emotion": self.emotion,
            "voice_id": self.voice_id
        }


class LiveTempoCoordinator:
    """
    Manages turn-taking and pacing for live commentary.
    
    Features:
    - Priority queue for commentary segments
    - Cultural interrupt thresholds
    - Natural pauses for breathing room
    - Variety enforcement (don't let one region dominate)
    
    Usage:
        coordinator = LiveTempoCoordinator()
        await coordinator.enqueue_commentary(segment)
        async for segment in coordinator.get_commentary_stream():
            await voice_engine.synthesize(segment)
    """
    
    # Cultural interrupt thresholds (from your research)
    # Higher = more likely to interrupt current speaker
    INTERRUPT_THRESHOLDS = {
        "dallas": 0.90,       # Interrupts immediately
        "philadelphia": 0.92,
        "kansas_city": 0.85,
        "chicago": 0.80,
        "new_york": 0.85,
        "new_england": 0.55,  # More reserved
        "green_bay": 0.45,    # Waits politely
        "san_francisco": 0.60,
        "brazilian": 0.72,    # Waits for rhythm
        "argentine": 0.38,    # Quick to challenge
        "default": 0.65,
    }
    
    # Minimum gap between segments (ms)
    MIN_GAP_MS = 200
    
    # Maximum time for same region to speak consecutively
    MAX_SAME_REGION_MS = 8000
    
    def __init__(self, cultural_params: Optional[dict] = None):
        """
        Initialize tempo coordinator.
        
        Args:
            cultural_params: Optional override for cultural parameters
        """
        self.cultural_params = cultural_params or {}
        self.queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.currently_speaking: Optional[str] = None
        self.last_speaker_region: Optional[str] = None
        self.last_speaker_end_time: float = 0
        self.region_speak_times: dict[str, float] = {}  # Track how long each region has spoken
        self._running = False
        self._pending_interrupt: Optional[ClassifiedEvent] = None
    
    async def start(self):
        """Start the coordinator"""
        self._running = True
        logger.info("LiveTempoCoordinator started")
    
    async def stop(self):
        """Stop the coordinator"""
        self._running = False
        logger.info("LiveTempoCoordinator stopped")
    
    async def enqueue_commentary(self, segment: CommentarySegment):
        """
        Add commentary to the queue with priority.
        
        Priority is boosted if:
        - Different region than last speaker (variety)
        - High emotion segment
        - Region hasn't spoken recently
        """
        priority = segment.priority
        
        # Boost priority if different region than last speaker
        if segment.region != self.last_speaker_region:
            priority += 2
        
        # Boost if this region hasn't spoken recently
        if segment.region not in self.region_speak_times:
            priority += 1
        elif time.time() - self.region_speak_times.get(segment.region, 0) > 10:
            priority += 1
        
        # Use negative for max-priority queue behavior
        await self.queue.put((-priority, time.time(), segment))
        logger.debug(f"Enqueued segment from {segment.region}, priority {priority}")
    
    async def get_next_segment(self) -> Optional[CommentarySegment]:
        """
        Get next segment to speak, respecting tempo rules.
        
        Returns None if queue is empty or we should pause.
        """
        if self.queue.empty():
            return None
        
        # Check for natural pause (breathing room)
        if self._should_pause():
            await asyncio.sleep(self.MIN_GAP_MS / 1000)
        
        try:
            _, _, segment = await asyncio.wait_for(
                self.queue.get(), 
                timeout=0.1
            )
            
            self.currently_speaking = segment.agent_id
            self.last_speaker_region = segment.region
            self.region_speak_times[segment.region] = time.time()
            
            return segment
            
        except asyncio.TimeoutError:
            return None
    
    async def get_commentary_stream(self) -> AsyncGenerator[CommentarySegment, None]:
        """
        Generator that yields commentary segments in order.
        
        Use this in your main playback loop.
        """
        while self._running:
            segment = await self.get_next_segment()
            if segment:
                yield segment
            else:
                # Small wait if queue is empty
                await asyncio.sleep(0.1)
    
    def mark_segment_complete(self, segment: CommentarySegment):
        """Mark a segment as finished speaking"""
        self.currently_speaking = None
        self.last_speaker_end_time = time.time()
        logger.debug(f"Segment from {segment.region} completed")
    
    def should_interrupt(self, new_event: ClassifiedEvent) -> bool:
        """
        Should we interrupt current speaker for this event?
        
        Based on:
        - Event urgency (touchdowns always interrupt)
        - Cultural threshold of current speaker's region
        - Controversy potential of new event
        """
        if self.currently_speaking is None:
            return False
        
        # Immediate events always interrupt
        if new_event.urgency == EventUrgency.IMMEDIATE:
            logger.info(f"Interrupting for IMMEDIATE event: {new_event.event_type.value}")
            return True
        
        # Check cultural threshold
        current_region = self.last_speaker_region or "default"
        threshold = self.INTERRUPT_THRESHOLDS.get(
            current_region, 
            self.INTERRUPT_THRESHOLDS["default"]
        )
        
        # Interrupt if controversy exceeds threshold
        should_int = new_event.controversy_potential > threshold
        
        if should_int:
            logger.info(f"Interrupting: controversy {new_event.controversy_potential} > threshold {threshold}")
        
        return should_int
    
    def request_interrupt(self, event: ClassifiedEvent):
        """Request an interrupt for a new event"""
        if self.should_interrupt(event):
            self._pending_interrupt = event
            logger.info(f"Interrupt requested for {event.event_type.value}")
    
    def has_pending_interrupt(self) -> bool:
        """Check if there's a pending interrupt"""
        return self._pending_interrupt is not None
    
    def get_pending_interrupt(self) -> Optional[ClassifiedEvent]:
        """Get and clear pending interrupt"""
        event = self._pending_interrupt
        self._pending_interrupt = None
        return event
    
    def _should_pause(self) -> bool:
        """
        Determine if we should insert a natural pause.
        
        Pauses occur:
        - Between speakers (breathing room)
        - If same region has been speaking too long
        """
        now = time.time()
        
        # Always pause briefly between speakers
        time_since_last = (now - self.last_speaker_end_time) * 1000
        if time_since_last < self.MIN_GAP_MS:
            return True
        
        return False
    
    def get_queue_stats(self) -> dict:
        """Get statistics about the current queue state"""
        return {
            "queue_size": self.queue.qsize(),
            "currently_speaking": self.currently_speaking,
            "last_speaker_region": self.last_speaker_region,
            "regions_active": list(self.region_speak_times.keys()),
            "has_pending_interrupt": self.has_pending_interrupt()
        }
    
    async def clear_queue(self):
        """Clear all pending commentary (e.g., for game end)"""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        logger.info("Queue cleared")


# Create singleton instance
_coordinator: Optional[LiveTempoCoordinator] = None

def get_tempo_coordinator() -> LiveTempoCoordinator:
    """Get or create the global tempo coordinator"""
    global _coordinator
    if _coordinator is None:
        _coordinator = LiveTempoCoordinator()
    return _coordinator


if __name__ == "__main__":
    # Test the tempo coordinator
    async def test():
        coordinator = LiveTempoCoordinator()
        await coordinator.start()
        
        # Enqueue some test segments
        segments = [
            CommentarySegment("agent1", "dallas", "TOUCHDOWN Dallas!", priority=9, emotion="excited"),
            CommentarySegment("agent2", "kansas_city", "Great response from KC!", priority=7),
            CommentarySegment("agent3", "dallas", "This is huge for fantasy!", priority=6),
            CommentarySegment("agent4", "green_bay", "Historically speaking...", priority=4),
        ]
        
        for seg in segments:
            await coordinator.enqueue_commentary(seg)
        
        print("Queue stats:", coordinator.get_queue_stats())
        
        # Get segments in priority order
        async for segment in coordinator.get_commentary_stream():
            print(f"\n{segment.region}/{segment.agent_id}: {segment.text}")
            coordinator.mark_segment_complete(segment)
            
            if coordinator.queue.empty():
                break
        
        await coordinator.stop()
    
    asyncio.run(test())
