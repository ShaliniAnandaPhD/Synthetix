"""
Live Audio Queue Manager

Client-side audio queue management to prevent pile-up during rapid events.
Handles interruption, skip, and fast-forward for late events.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Any
from enum import Enum

logger = logging.getLogger(__name__)


class AudioPriority(Enum):
    """Priority levels for audio queue"""
    IMMEDIATE = 1   # Touchdowns, turnovers - interrupt current
    HIGH = 2        # Big plays - queue next
    NORMAL = 3      # Regular commentary
    LOW = 4         # Background/filler


class QueueAction(Enum):
    """Actions for queue items"""
    PLAY = "play"
    SKIP = "skip"
    INTERRUPT = "interrupt"


@dataclass
class AudioQueueItem:
    """An audio item in the queue"""
    item_id: str
    audio_data: str  # base64 encoded
    text: str
    region: str
    agent_type: str
    priority: AudioPriority
    duration_ms: int
    created_at: float = field(default_factory=time.time)
    event_timestamp: float = 0  # When the event occurred
    max_age_ms: int = 5000  # Skip if older than this
    
    @property
    def age_ms(self) -> int:
        """How old is this item"""
        return int((time.time() - self.created_at) * 1000)
    
    @property
    def is_stale(self) -> bool:
        """Should this item be skipped due to age"""
        return self.age_ms > self.max_age_ms


class LiveAudioQueue:
    """
    Manages audio playback queue for live commentary.
    
    Features:
    - Priority-based ordering
    - Stale item skipping
    - Interruption for high-priority events
    - Rate limiting to prevent pile-up
    
    Usage (client-side):
        queue = LiveAudioQueue(on_play=audio_player.play)
        
        # Add commentary as it arrives
        queue.add(AudioQueueItem(...))
        
        # Start processing
        await queue.start()
    """
    
    # Maximum queue size
    MAX_QUEUE_SIZE = 10
    
    # Minimum gap between audio (ms)
    MIN_GAP_MS = 200
    
    def __init__(
        self,
        on_play: Optional[Callable[[AudioQueueItem], Any]] = None,
        on_skip: Optional[Callable[[AudioQueueItem, str], Any]] = None,
        on_interrupt: Optional[Callable[[], Any]] = None,
    ):
        """
        Initialize audio queue.
        
        Args:
            on_play: Callback when audio should play
            on_skip: Callback when audio is skipped (item, reason)
            on_interrupt: Callback to stop current audio
        """
        self.on_play = on_play
        self.on_skip = on_skip
        self.on_interrupt = on_interrupt
        
        self._queue: List[AudioQueueItem] = []
        self._current_item: Optional[AudioQueueItem] = None
        self._is_playing = False
        self._running = False
        self._lock = asyncio.Lock()
        
        self._stats = {
            "played": 0,
            "skipped_stale": 0,
            "skipped_overflow": 0,
            "interrupted": 0,
        }
    
    async def start(self):
        """Start the queue processor"""
        self._running = True
        asyncio.create_task(self._process_loop())
        logger.info("Audio queue started")
    
    async def stop(self):
        """Stop the queue processor"""
        self._running = False
        self._queue.clear()
        logger.info("Audio queue stopped")
    
    async def add(self, item: AudioQueueItem) -> QueueAction:
        """
        Add an item to the queue.
        
        Returns the action taken.
        """
        async with self._lock:
            # Check if this should interrupt current playback
            if self._should_interrupt(item):
                await self._interrupt()
                self._queue.insert(0, item)
                return QueueAction.INTERRUPT
            
            # Check queue overflow
            if len(self._queue) >= self.MAX_QUEUE_SIZE:
                # Remove lowest priority item
                self._queue.sort(key=lambda x: x.priority.value)
                removed = self._queue.pop()
                self._stats["skipped_overflow"] += 1
                if self.on_skip:
                    self.on_skip(removed, "overflow")
            
            # Insert by priority
            insert_idx = 0
            for i, existing in enumerate(self._queue):
                if item.priority.value <= existing.priority.value:
                    insert_idx = i
                    break
                insert_idx = i + 1
            
            self._queue.insert(insert_idx, item)
            return QueueAction.PLAY
    
    async def clear(self):
        """Clear all pending items"""
        async with self._lock:
            self._queue.clear()
    
    async def skip_current(self):
        """Skip currently playing audio"""
        if self._current_item and self.on_interrupt:
            self.on_interrupt()
            self._is_playing = False
    
    def get_queue_length(self) -> int:
        """Get number of items in queue"""
        return len(self._queue)
    
    def get_stats(self) -> dict:
        """Get queue statistics"""
        return {
            **self._stats,
            "queue_length": len(self._queue),
            "is_playing": self._is_playing,
        }
    
    def _should_interrupt(self, item: AudioQueueItem) -> bool:
        """Check if new item should interrupt current playback"""
        if not self._is_playing or not self._current_item:
            return False
        
        # Only IMMEDIATE priority items can interrupt
        if item.priority != AudioPriority.IMMEDIATE:
            return False
        
        # Don't interrupt another IMMEDIATE
        if self._current_item.priority == AudioPriority.IMMEDIATE:
            return False
        
        return True
    
    async def _interrupt(self):
        """Interrupt current playback"""
        if self.on_interrupt:
            self.on_interrupt()
        
        self._is_playing = False
        self._stats["interrupted"] += 1
        logger.info("Audio interrupted for higher priority")
    
    async def _process_loop(self):
        """Main processing loop"""
        while self._running:
            try:
                if not self._is_playing and self._queue:
                    async with self._lock:
                        if self._queue:
                            item = self._queue.pop(0)
                            await self._play_item(item)
                
                await asyncio.sleep(0.05)  # 50ms poll
                
            except Exception as e:
                logger.error(f"Queue processing error: {e}")
                await asyncio.sleep(0.1)
    
    async def _play_item(self, item: AudioQueueItem):
        """Play an audio item"""
        # Check if stale
        if item.is_stale:
            self._stats["skipped_stale"] += 1
            if self.on_skip:
                self.on_skip(item, "stale")
            logger.debug(f"Skipped stale audio ({item.age_ms}ms old)")
            return
        
        # Play audio
        self._current_item = item
        self._is_playing = True
        self._stats["played"] += 1
        
        if self.on_play:
            self.on_play(item)
        
        # Wait for duration + gap
        await asyncio.sleep((item.duration_ms + self.MIN_GAP_MS) / 1000)
        
        self._is_playing = False
        self._current_item = None


# TypeScript equivalent for frontend
TYPESCRIPT_QUEUE = '''
// LiveAudioQueue.ts - Client-side audio queue

interface AudioQueueItem {
  itemId: string;
  audioData: string;  // base64
  text: string;
  region: string;
  agentType: string;
  priority: 'immediate' | 'high' | 'normal' | 'low';
  durationMs: number;
  createdAt: number;
  maxAgeMs: number;
}

class LiveAudioQueue {
  private queue: AudioQueueItem[] = [];
  private currentAudio: HTMLAudioElement | null = null;
  private isPlaying = false;
  private maxQueueSize = 10;
  
  constructor(private audioContext?: AudioContext) {}
  
  add(item: AudioQueueItem): 'play' | 'skip' | 'interrupt' {
    const now = Date.now();
    
    // Check for interruption
    if (this.shouldInterrupt(item)) {
      this.interrupt();
      this.queue.unshift(item);
      this.playNext();
      return 'interrupt';
    }
    
    // Check overflow
    if (this.queue.length >= this.maxQueueSize) {
      this.queue.pop();  // Remove last (lowest priority)
    }
    
    // Insert by priority
    const priorityOrder = { immediate: 0, high: 1, normal: 2, low: 3 };
    const insertIdx = this.queue.findIndex(
      q => priorityOrder[item.priority] < priorityOrder[q.priority]
    );
    
    if (insertIdx === -1) {
      this.queue.push(item);
    } else {
      this.queue.splice(insertIdx, 0, item);
    }
    
    if (!this.isPlaying) {
      this.playNext();
    }
    
    return 'play';
  }
  
  private shouldInterrupt(item: AudioQueueItem): boolean {
    return this.isPlaying && item.priority === 'immediate';
  }
  
  private interrupt() {
    if (this.currentAudio) {
      this.currentAudio.pause();
      this.currentAudio = null;
    }
    this.isPlaying = false;
  }
  
  private async playNext() {
    if (this.queue.length === 0 || this.isPlaying) return;
    
    const item = this.queue.shift()!;
    const age = Date.now() - item.createdAt;
    
    // Skip stale items
    if (age > item.maxAgeMs) {
      console.log(`Skipping stale audio (${age}ms old)`);
      this.playNext();
      return;
    }
    
    this.isPlaying = true;
    
    try {
      const audio = new Audio(`data:audio/wav;base64,${item.audioData}`);
      this.currentAudio = audio;
      
      audio.onended = () => {
        this.isPlaying = false;
        this.currentAudio = null;
        setTimeout(() => this.playNext(), 200);  // 200ms gap
      };
      
      audio.onerror = () => {
        this.isPlaying = false;
        this.currentAudio = null;
        this.playNext();
      };
      
      await audio.play();
    } catch (e) {
      console.error('Audio playback error:', e);
      this.isPlaying = false;
      this.playNext();
    }
  }
  
  clear() {
    this.queue = [];
    this.interrupt();
  }
  
  getQueueLength(): number {
    return this.queue.length;
  }
}

export { LiveAudioQueue, AudioQueueItem };
'''


if __name__ == "__main__":
    async def test():
        played = []
        skipped = []
        
        queue = LiveAudioQueue(
            on_play=lambda item: played.append(item.text),
            on_skip=lambda item, reason: skipped.append((item.text, reason)),
        )
        
        await queue.start()
        
        # Add items
        await queue.add(AudioQueueItem(
            item_id="1", audio_data="", text="Normal play",
            region="kc", agent_type="analyst",
            priority=AudioPriority.NORMAL, duration_ms=100
        ))
        
        await queue.add(AudioQueueItem(
            item_id="2", audio_data="", text="TOUCHDOWN!",
            region="kc", agent_type="homer",
            priority=AudioPriority.IMMEDIATE, duration_ms=100
        ))
        
        await asyncio.sleep(0.5)
        
        print(f"Played: {played}")
        print(f"Skipped: {skipped}")
        print(f"Stats: {queue.get_stats()}")
        
        await queue.stop()
    
    asyncio.run(test())
