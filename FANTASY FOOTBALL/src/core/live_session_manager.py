"""
Live Session Manager

Manages active live commentary sessions with:
- Session lifecycle (create, track, cleanup)
- Reconnection handling
- Metrics logging
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Set, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class SessionState(Enum):
    """Session lifecycle states"""
    CONNECTING = "connecting"
    ACTIVE = "active"
    PAUSED = "paused"
    RECONNECTING = "reconnecting"
    ENDED = "ended"


@dataclass
class SessionMetrics:
    """Metrics for a live session"""
    events_received: int = 0
    commentary_sent: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    fallbacks_used: int = 0
    avg_latency_ms: float = 0
    latency_samples: int = 0
    errors: int = 0
    
    def record_latency(self, latency_ms: int):
        """Update running average latency"""
        self.latency_samples += 1
        self.avg_latency_ms = (
            (self.avg_latency_ms * (self.latency_samples - 1) + latency_ms)
            / self.latency_samples
        )
    
    def to_dict(self) -> dict:
        return {
            "events_received": self.events_received,
            "commentary_sent": self.commentary_sent,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "fallbacks_used": self.fallbacks_used,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "errors": self.errors,
        }


@dataclass
class LiveSession:
    """An active live commentary session"""
    session_id: str
    creator_id: str
    game_id: str
    regions: list
    websocket: Any = None
    state: SessionState = SessionState.CONNECTING
    started_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    metrics: SessionMetrics = field(default_factory=SessionMetrics)
    reconnect_token: Optional[str] = None
    
    @property
    def duration_seconds(self) -> int:
        return int(time.time() - self.started_at)
    
    @property
    def idle_seconds(self) -> int:
        return int(time.time() - self.last_activity)
    
    def touch(self):
        """Update last activity time"""
        self.last_activity = time.time()
    
    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "creator_id": self.creator_id,
            "game_id": self.game_id,
            "regions": self.regions,
            "state": self.state.value,
            "duration_seconds": self.duration_seconds,
            "idle_seconds": self.idle_seconds,
            "metrics": self.metrics.to_dict(),
        }


class LiveSessionManager:
    """
    Manages all active live commentary sessions.
    
    Features:
    - Track sessions per game
    - Handle reconnection with state recovery
    - Log metrics to monitoring
    - Clean up idle sessions
    
    Usage:
        manager = LiveSessionManager()
        session = await manager.create_session(creator_id, game_id, regions, websocket)
        
        # During game
        session.metrics.events_received += 1
        session.touch()
        
        # Cleanup
        await manager.end_session(session.session_id)
    """
    
    # Session timeout (5 minutes of inactivity)
    IDLE_TIMEOUT_SECONDS = 300
    
    # Maximum sessions per game
    MAX_SESSIONS_PER_GAME = 100
    
    def __init__(self, metrics_logger=None):
        """
        Initialize session manager.
        
        Args:
            metrics_logger: Optional W&B or other metrics logger
        """
        self.sessions: Dict[str, LiveSession] = {}
        self.game_sessions: Dict[str, Set[str]] = {}  # game_id -> session_ids
        self.reconnect_tokens: Dict[str, str] = {}  # token -> session_id
        self.metrics_logger = metrics_logger
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the session manager and cleanup task"""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("LiveSessionManager started")
    
    async def stop(self):
        """Stop the session manager"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # End all sessions
        for session_id in list(self.sessions.keys()):
            await self.end_session(session_id)
        
        logger.info("LiveSessionManager stopped")
    
    async def create_session(
        self, 
        creator_id: str, 
        game_id: str, 
        regions: list,
        websocket: Any = None
    ) -> LiveSession:
        """
        Create a new live session.
        
        Returns the new session or raises if limit exceeded.
        """
        # Check game session limit
        current_game_sessions = len(self.game_sessions.get(game_id, set()))
        if current_game_sessions >= self.MAX_SESSIONS_PER_GAME:
            raise ValueError(f"Game {game_id} has reached max sessions ({self.MAX_SESSIONS_PER_GAME})")
        
        # Generate session ID and reconnect token
        session_id = f"{creator_id}_{game_id}_{int(time.time() * 1000)}"
        reconnect_token = self._generate_reconnect_token()
        
        session = LiveSession(
            session_id=session_id,
            creator_id=creator_id,
            game_id=game_id,
            regions=regions,
            websocket=websocket,
            state=SessionState.ACTIVE,
            reconnect_token=reconnect_token
        )
        
        self.sessions[session_id] = session
        
        if game_id not in self.game_sessions:
            self.game_sessions[game_id] = set()
        self.game_sessions[game_id].add(session_id)
        
        self.reconnect_tokens[reconnect_token] = session_id
        
        # Log to metrics
        self._log_event("session_created", session)
        
        logger.info(f"Session created: {session_id} for game {game_id}")
        return session
    
    async def end_session(self, session_id: str) -> Optional[dict]:
        """
        End a session and return final stats.
        """
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        session.state = SessionState.ENDED
        
        # Clean up references
        del self.sessions[session_id]
        
        if session.game_id in self.game_sessions:
            self.game_sessions[session.game_id].discard(session_id)
            if not self.game_sessions[session.game_id]:
                del self.game_sessions[session.game_id]
        
        if session.reconnect_token:
            self.reconnect_tokens.pop(session.reconnect_token, None)
        
        # Log final stats
        stats = session.to_dict()
        self._log_event("session_ended", session)
        
        logger.info(f"Session ended: {session_id} after {session.duration_seconds}s")
        return stats
    
    async def reconnect_session(
        self, 
        reconnect_token: str,
        new_websocket: Any
    ) -> Optional[LiveSession]:
        """
        Reconnect to an existing session using token.
        """
        session_id = self.reconnect_tokens.get(reconnect_token)
        if not session_id or session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        session.websocket = new_websocket
        session.state = SessionState.ACTIVE
        session.touch()
        
        self._log_event("session_reconnected", session)
        
        logger.info(f"Session reconnected: {session_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[LiveSession]:
        """Get a session by ID"""
        return self.sessions.get(session_id)
    
    def get_game_sessions(self, game_id: str) -> list[LiveSession]:
        """Get all sessions for a game"""
        session_ids = self.game_sessions.get(game_id, set())
        return [self.sessions[sid] for sid in session_ids if sid in self.sessions]
    
    async def broadcast_to_game(self, game_id: str, message: dict):
        """Send message to all sessions watching a game"""
        sessions = self.get_game_sessions(game_id)
        
        for session in sessions:
            if session.websocket and session.state == SessionState.ACTIVE:
                try:
                    if hasattr(session.websocket, 'send_json'):
                        await session.websocket.send_json(message)
                    elif hasattr(session.websocket, 'send'):
                        import json
                        await session.websocket.send(json.dumps(message))
                    
                    session.metrics.commentary_sent += 1
                    session.touch()
                    
                except Exception as e:
                    logger.warning(f"Broadcast failed for {session.session_id}: {e}")
                    session.state = SessionState.RECONNECTING
    
    def get_stats(self) -> dict:
        """Get overall session statistics"""
        total_events = sum(s.metrics.events_received for s in self.sessions.values())
        total_commentary = sum(s.metrics.commentary_sent for s in self.sessions.values())
        
        return {
            "active_sessions": len(self.sessions),
            "games_active": len(self.game_sessions),
            "total_events_processed": total_events,
            "total_commentary_sent": total_commentary,
            "sessions_by_game": {
                gid: len(sids) 
                for gid, sids in self.game_sessions.items()
            },
        }
    
    async def _cleanup_loop(self):
        """Background task to clean up idle sessions"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                now = time.time()
                idle_sessions = [
                    sid for sid, session in self.sessions.items()
                    if session.idle_seconds > self.IDLE_TIMEOUT_SECONDS
                ]
                
                for session_id in idle_sessions:
                    logger.info(f"Cleaning up idle session: {session_id}")
                    await self.end_session(session_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    def _generate_reconnect_token(self) -> str:
        """Generate a unique reconnect token"""
        import secrets
        return secrets.token_urlsafe(32)
    
    def _log_event(self, event_type: str, session: LiveSession):
        """Log session event to metrics"""
        if self.metrics_logger:
            try:
                self.metrics_logger.log({
                    f"live_session_{event_type}": 1,
                    "session_id": session.session_id,
                    "game_id": session.game_id,
                    "duration_seconds": session.duration_seconds,
                    **session.metrics.to_dict()
                })
            except Exception as e:
                logger.warning(f"Metrics logging failed: {e}")


# Singleton instance
_session_manager: Optional[LiveSessionManager] = None

def get_session_manager() -> LiveSessionManager:
    """Get or create the global session manager"""
    global _session_manager
    if _session_manager is None:
        _session_manager = LiveSessionManager()
    return _session_manager


if __name__ == "__main__":
    async def test():
        manager = LiveSessionManager()
        await manager.start()
        
        # Create sessions
        s1 = await manager.create_session("creator1", "KC_BUF", ["kansas_city"])
        s2 = await manager.create_session("creator2", "KC_BUF", ["buffalo"])
        
        print(f"\nActive sessions: {manager.get_stats()}")
        
        # Simulate activity
        s1.metrics.events_received += 5
        s1.metrics.commentary_sent += 10
        s1.touch()
        
        # Reconnect test
        token = s1.reconnect_token
        s1_reconnected = await manager.reconnect_session(token, None)
        print(f"\nReconnected: {s1_reconnected.session_id}")
        
        # End sessions
        stats = await manager.end_session(s1.session_id)
        print(f"\nSession stats: {stats}")
        
        await manager.stop()
    
    asyncio.run(test())
