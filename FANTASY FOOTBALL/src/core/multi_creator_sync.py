"""
Multi-Creator Sync

Shares event processing across multiple creators watching the same game.
Efficient broadcast with regional filtering.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional, Any, Callable
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class CreatorConnection:
    """A connected creator"""
    creator_id: str
    regions: List[str]
    websocket: Any
    connected_at: float = field(default_factory=time.time)
    messages_received: int = 0


@dataclass
class GameBroadcast:
    """A game being broadcast to multiple creators"""
    game_id: str
    home_team: str
    away_team: str
    creators: Dict[str, CreatorConnection] = field(default_factory=dict)
    events_processed: int = 0
    started_at: float = field(default_factory=time.time)


class MultiCreatorSync:
    """
    Synchronizes live commentary across multiple creators watching the same game.
    
    Features:
    - Single event processing per game (shared)
    - Regional filtering per creator
    - Efficient broadcast to all connected clients
    
    Usage:
        sync = MultiCreatorSync()
        
        # When creator connects
        await sync.join_game("KC_BUF", creator_connection)
        
        # When event occurs (called once per event, not per creator)
        await sync.broadcast_event("KC_BUF", event_data)
        
        # When commentary is generated
        await sync.broadcast_commentary("KC_BUF", commentary)
    """
    
    def __init__(self):
        self.games: Dict[str, GameBroadcast] = {}
        self._event_handlers: Dict[str, Callable] = {}
        self._lock = asyncio.Lock()
    
    async def start_game(
        self, 
        game_id: str, 
        home_team: str, 
        away_team: str
    ) -> GameBroadcast:
        """Start tracking a new game"""
        async with self._lock:
            if game_id in self.games:
                return self.games[game_id]
            
            game = GameBroadcast(
                game_id=game_id,
                home_team=home_team,
                away_team=away_team
            )
            self.games[game_id] = game
            logger.info(f"Started game broadcast: {game_id}")
            return game
    
    async def end_game(self, game_id: str) -> Optional[dict]:
        """End a game, disconnect all creators"""
        async with self._lock:
            if game_id not in self.games:
                return None
            
            game = self.games[game_id]
            stats = {
                "game_id": game_id,
                "creators_served": len(game.creators),
                "events_processed": game.events_processed,
                "duration_seconds": int(time.time() - game.started_at)
            }
            
            # Notify all creators game is ending
            await self._broadcast_to_game(game_id, {
                "type": "game_end",
                "stats": stats
            })
            
            del self.games[game_id]
            logger.info(f"Ended game broadcast: {game_id}")
            return stats
    
    async def join_game(
        self,
        game_id: str,
        creator_id: str,
        regions: List[str],
        websocket: Any
    ) -> bool:
        """Add a creator to a game broadcast"""
        if game_id not in self.games:
            return False
        
        connection = CreatorConnection(
            creator_id=creator_id,
            regions=regions,
            websocket=websocket
        )
        
        self.games[game_id].creators[creator_id] = connection
        
        # Send join confirmation
        await self._send_to_creator(websocket, {
            "type": "joined_game",
            "game_id": game_id,
            "creators_watching": len(self.games[game_id].creators),
            "your_regions": regions
        })
        
        logger.info(f"Creator {creator_id} joined {game_id}")
        return True
    
    async def leave_game(self, game_id: str, creator_id: str):
        """Remove a creator from a game broadcast"""
        if game_id not in self.games:
            return
        
        if creator_id in self.games[game_id].creators:
            del self.games[game_id].creators[creator_id]
            logger.info(f"Creator {creator_id} left {game_id}")
    
    async def broadcast_event(self, game_id: str, event: dict):
        """
        Broadcast an event to all creators watching a game.
        Event is processed once, sent to all.
        """
        if game_id not in self.games:
            return
        
        self.games[game_id].events_processed += 1
        
        await self._broadcast_to_game(game_id, {
            "type": "event",
            "data": event,
            "timestamp": time.time()
        })
    
    async def broadcast_commentary(
        self, 
        game_id: str, 
        region: str,
        commentary: dict
    ):
        """
        Broadcast commentary with regional filtering.
        Only sends to creators who selected this region.
        """
        if game_id not in self.games:
            return
        
        game = self.games[game_id]
        
        for creator in game.creators.values():
            # Filter by region
            if region in creator.regions or "all" in creator.regions:
                await self._send_to_creator(creator.websocket, {
                    "type": "commentary",
                    "region": region,
                    **commentary
                })
                creator.messages_received += 1
    
    async def _broadcast_to_game(self, game_id: str, message: dict):
        """Broadcast to all creators in a game"""
        if game_id not in self.games:
            return
        
        game = self.games[game_id]
        failed = []
        
        for creator_id, creator in game.creators.items():
            try:
                await self._send_to_creator(creator.websocket, message)
                creator.messages_received += 1
            except Exception as e:
                logger.warning(f"Broadcast failed for {creator_id}: {e}")
                failed.append(creator_id)
        
        # Clean up failed connections
        for creator_id in failed:
            await self.leave_game(game_id, creator_id)
    
    async def _send_to_creator(self, websocket: Any, message: dict):
        """Send message to a creator's websocket"""
        if websocket is None:
            return
        
        if hasattr(websocket, 'send_json'):
            await websocket.send_json(message)
        elif hasattr(websocket, 'send'):
            import json
            await websocket.send(json.dumps(message))
    
    def get_game_stats(self, game_id: str) -> Optional[dict]:
        """Get stats for a game"""
        if game_id not in self.games:
            return None
        
        game = self.games[game_id]
        return {
            "game_id": game_id,
            "home_team": game.home_team,
            "away_team": game.away_team,
            "creators_connected": len(game.creators),
            "events_processed": game.events_processed,
            "uptime_seconds": int(time.time() - game.started_at),
            "creators": [
                {
                    "creator_id": c.creator_id,
                    "regions": c.regions,
                    "messages_received": c.messages_received
                }
                for c in game.creators.values()
            ]
        }
    
    def get_all_games(self) -> List[dict]:
        """Get stats for all active games"""
        return [
            {
                "game_id": gid,
                "creators": len(game.creators),
                "events": game.events_processed
            }
            for gid, game in self.games.items()
        ]


# Singleton
_sync: Optional[MultiCreatorSync] = None

def get_multi_creator_sync() -> MultiCreatorSync:
    global _sync
    if _sync is None:
        _sync = MultiCreatorSync()
    return _sync
