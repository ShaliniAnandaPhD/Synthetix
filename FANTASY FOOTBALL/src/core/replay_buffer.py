"""
Replay Buffer

Stores recent commentary for late joiners to catch up.
Maintains a rolling window of recent events and commentary.
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class ReplayItem:
    """A single item in the replay buffer"""
    item_type: str  # "event", "commentary", "score_update"
    data: dict
    timestamp: float = field(default_factory=time.time)
    region: str = ""


@dataclass
class GameReplayBuffer:
    """Replay buffer for a single game"""
    game_id: str
    items: deque = field(default_factory=lambda: deque(maxlen=100))
    max_age_seconds: int = 300  # 5 minutes
    
    def add(self, item: ReplayItem):
        """Add an item to the buffer"""
        self.items.append(item)
    
    def get_recent(
        self, 
        max_items: int = 20,
        max_age_seconds: Optional[int] = None,
        regions: Optional[List[str]] = None
    ) -> List[ReplayItem]:
        """
        Get recent items from buffer.
        
        Args:
            max_items: Maximum items to return
            max_age_seconds: Only items newer than this
            regions: Filter by regions (if specified)
        """
        cutoff = time.time() - (max_age_seconds or self.max_age_seconds)
        
        result = []
        for item in reversed(self.items):
            if item.timestamp < cutoff:
                break
            
            if regions and item.region and item.region not in regions:
                continue
            
            result.append(item)
            
            if len(result) >= max_items:
                break
        
        return list(reversed(result))
    
    def clear(self):
        """Clear the buffer"""
        self.items.clear()


class ReplayBufferManager:
    """
    Manages replay buffers for all active games.
    
    Features:
    - Store recent 5 minutes of commentary
    - New joiners can catch up
    - Memory-efficient with rolling window
    
    Usage:
        manager = ReplayBufferManager()
        
        # Store items during game
        manager.add_commentary("KC_BUF", "kansas_city", {...})
        manager.add_event("KC_BUF", {...})
        
        # When new creator joins
        recent = manager.get_catchup("KC_BUF", ["kansas_city"], max_items=10)
    """
    
    def __init__(
        self,
        max_items_per_game: int = 100,
        max_age_seconds: int = 300
    ):
        self.max_items = max_items_per_game
        self.max_age = max_age_seconds
        self.buffers: Dict[str, GameReplayBuffer] = {}
    
    def get_or_create_buffer(self, game_id: str) -> GameReplayBuffer:
        """Get or create a buffer for a game"""
        if game_id not in self.buffers:
            self.buffers[game_id] = GameReplayBuffer(
                game_id=game_id,
                items=deque(maxlen=self.max_items),
                max_age_seconds=self.max_age
            )
        return self.buffers[game_id]
    
    def add_event(self, game_id: str, event_data: dict):
        """Add an event to the replay buffer"""
        buffer = self.get_or_create_buffer(game_id)
        buffer.add(ReplayItem(
            item_type="event",
            data=event_data
        ))
    
    def add_commentary(
        self, 
        game_id: str, 
        region: str,
        commentary_data: dict
    ):
        """Add commentary to the replay buffer"""
        buffer = self.get_or_create_buffer(game_id)
        buffer.add(ReplayItem(
            item_type="commentary",
            data=commentary_data,
            region=region
        ))
    
    def add_score_update(self, game_id: str, score_data: dict):
        """Add a score update"""
        buffer = self.get_or_create_buffer(game_id)
        buffer.add(ReplayItem(
            item_type="score_update",
            data=score_data
        ))
    
    def get_catchup(
        self,
        game_id: str,
        regions: Optional[List[str]] = None,
        max_items: int = 20,
        max_age_seconds: Optional[int] = None
    ) -> List[dict]:
        """
        Get recent items for a late-joining creator.
        
        Returns items formatted for sending over WebSocket.
        """
        if game_id not in self.buffers:
            return []
        
        buffer = self.buffers[game_id]
        items = buffer.get_recent(
            max_items=max_items,
            max_age_seconds=max_age_seconds,
            regions=regions
        )
        
        return [
            {
                "type": f"replay_{item.item_type}",
                "data": item.data,
                "region": item.region,
                "timestamp": item.timestamp,
                "is_replay": True
            }
            for item in items
        ]
    
    def remove_game(self, game_id: str):
        """Remove a game's buffer"""
        if game_id in self.buffers:
            del self.buffers[game_id]
    
    def get_stats(self) -> dict:
        """Get buffer statistics"""
        return {
            "active_games": len(self.buffers),
            "total_items": sum(len(b.items) for b in self.buffers.values()),
            "by_game": {
                gid: len(b.items)
                for gid, b in self.buffers.items()
            }
        }


# Singleton
_replay_manager: Optional[ReplayBufferManager] = None

def get_replay_buffer() -> ReplayBufferManager:
    global _replay_manager
    if _replay_manager is None:
        _replay_manager = ReplayBufferManager()
    return _replay_manager
