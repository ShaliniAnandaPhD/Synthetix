"""
Rate Limiting

Per-creator, per-game limits to prevent runaway costs.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    requests_per_minute: int = 60
    requests_per_hour: int = 500
    requests_per_day: int = 5000
    debates_per_day: int = 50
    live_sessions_per_day: int = 10


@dataclass
class RateLimitState:
    """Current state for a rate limit key"""
    minute_count: int = 0
    minute_start: float = 0
    hour_count: int = 0
    hour_start: float = 0
    day_count: int = 0
    day_start: float = 0
    debates_today: int = 0
    live_sessions_today: int = 0


class RateLimiter:
    """
    Rate limiter for API requests and resource usage.
    
    Usage:
        limiter = RateLimiter()
        
        # Check before processing
        if limiter.is_allowed("creator123", "request"):
            process_request()
        else:
            return 429  # Too Many Requests
        
        # Record usage
        limiter.record_request("creator123")
    """
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self._states: Dict[str, RateLimitState] = defaultdict(RateLimitState)
    
    def is_allowed(self, key: str, action: str = "request") -> bool:
        """Check if action is allowed for this key"""
        state = self._get_state(key)
        
        if action == "request":
            return self._check_request_limits(state)
        elif action == "debate":
            return state.debates_today < self.config.debates_per_day
        elif action == "live_session":
            return state.live_sessions_today < self.config.live_sessions_per_day
        
        return True
    
    def record_request(self, key: str):
        """Record a request"""
        state = self._get_state(key)
        state.minute_count += 1
        state.hour_count += 1
        state.day_count += 1
    
    def record_debate(self, key: str):
        """Record a debate creation"""
        state = self._get_state(key)
        state.debates_today += 1
    
    def record_live_session(self, key: str):
        """Record a live session"""
        state = self._get_state(key)
        state.live_sessions_today += 1
    
    def get_limits(self, key: str) -> dict:
        """Get current limit status for a key"""
        state = self._get_state(key)
        
        return {
            "key": key,
            "minute": {
                "used": state.minute_count,
                "limit": self.config.requests_per_minute,
                "remaining": max(0, self.config.requests_per_minute - state.minute_count)
            },
            "hour": {
                "used": state.hour_count,
                "limit": self.config.requests_per_hour,
                "remaining": max(0, self.config.requests_per_hour - state.hour_count)
            },
            "day": {
                "used": state.day_count,
                "limit": self.config.requests_per_day,
                "remaining": max(0, self.config.requests_per_day - state.day_count)
            },
            "debates_today": {
                "used": state.debates_today,
                "limit": self.config.debates_per_day,
                "remaining": max(0, self.config.debates_per_day - state.debates_today)
            },
            "live_sessions_today": {
                "used": state.live_sessions_today,
                "limit": self.config.live_sessions_per_day,
                "remaining": max(0, self.config.live_sessions_per_day - state.live_sessions_today)
            }
        }
    
    def reset(self, key: str):
        """Reset limits for a key"""
        if key in self._states:
            del self._states[key]
    
    def _get_state(self, key: str) -> RateLimitState:
        """Get or create state, resetting expired windows"""
        now = time.time()
        state = self._states[key]
        
        # Reset minute window
        if now - state.minute_start >= 60:
            state.minute_count = 0
            state.minute_start = now
        
        # Reset hour window
        if now - state.hour_start >= 3600:
            state.hour_count = 0
            state.hour_start = now
        
        # Reset day window
        if now - state.day_start >= 86400:
            state.day_count = 0
            state.debates_today = 0
            state.live_sessions_today = 0
            state.day_start = now
        
        return state
    
    def _check_request_limits(self, state: RateLimitState) -> bool:
        """Check all request rate limits"""
        if state.minute_count >= self.config.requests_per_minute:
            return False
        if state.hour_count >= self.config.requests_per_hour:
            return False
        if state.day_count >= self.config.requests_per_day:
            return False
        return True


class GameRateLimiter:
    """
    Rate limiter specific to live games.
    Prevents runaway costs during live broadcasts.
    """
    
    def __init__(
        self,
        events_per_minute: int = 30,
        commentary_per_minute: int = 60,
        tts_calls_per_minute: int = 20
    ):
        self.events_per_minute = events_per_minute
        self.commentary_per_minute = commentary_per_minute
        self.tts_calls_per_minute = tts_calls_per_minute
        
        self._game_states: Dict[str, dict] = defaultdict(lambda: {
            "events": [], "commentary": [], "tts": []
        })
    
    def is_event_allowed(self, game_id: str) -> bool:
        return self._check_rate(game_id, "events", self.events_per_minute)
    
    def is_commentary_allowed(self, game_id: str) -> bool:
        return self._check_rate(game_id, "commentary", self.commentary_per_minute)
    
    def is_tts_allowed(self, game_id: str) -> bool:
        return self._check_rate(game_id, "tts", self.tts_calls_per_minute)
    
    def record_event(self, game_id: str):
        self._record(game_id, "events")
    
    def record_commentary(self, game_id: str):
        self._record(game_id, "commentary")
    
    def record_tts(self, game_id: str):
        self._record(game_id, "tts")
    
    def get_game_stats(self, game_id: str) -> dict:
        state = self._game_states[game_id]
        now = time.time()
        cutoff = now - 60
        
        return {
            "game_id": game_id,
            "events_last_minute": len([t for t in state["events"] if t > cutoff]),
            "commentary_last_minute": len([t for t in state["commentary"] if t > cutoff]),
            "tts_last_minute": len([t for t in state["tts"] if t > cutoff]),
        }
    
    def _check_rate(self, game_id: str, category: str, limit: int) -> bool:
        state = self._game_states[game_id]
        now = time.time()
        cutoff = now - 60
        
        # Clean old entries
        state[category] = [t for t in state[category] if t > cutoff]
        
        return len(state[category]) < limit
    
    def _record(self, game_id: str, category: str):
        self._game_states[game_id][category].append(time.time())


# Singletons
_limiter: Optional[RateLimiter] = None
_game_limiter: Optional[GameRateLimiter] = None

def get_rate_limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter()
    return _limiter

def get_game_rate_limiter() -> GameRateLimiter:
    global _game_limiter
    if _game_limiter is None:
        _game_limiter = GameRateLimiter()
    return _game_limiter
