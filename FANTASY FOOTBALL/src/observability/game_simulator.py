"""
Mock Game Simulator

Replay recorded games at real-time speed for testing.
"""

import asyncio
import logging
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, AsyncGenerator
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class GameEvent:
    """A single game event"""
    event_id: str
    event_type: str
    description: str
    team: str
    timestamp: float  # Seconds from game start
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecordedGame:
    """A recorded game for replay"""
    game_id: str
    home_team: str
    away_team: str
    events: List[GameEvent] = field(default_factory=list)
    final_score: str = ""
    duration_seconds: float = 0


class MockGameSimulator:
    """
    Simulates NFL games for testing.
    
    Features:
    - Replay recorded games
    - Real-time or accelerated playback
    - Generate random events
    
    Usage:
        sim = MockGameSimulator()
        
        # Load recorded game
        sim.load_game("kc_buf_2024")
        
        # Replay at 10x speed
        async for event in sim.replay("kc_buf_2024", speed=10):
            process(event)
    """
    
    # Sample events for random generation
    SAMPLE_EVENTS = [
        ("play", "Run up the middle for 3 yards", 0.4),
        ("play", "Pass complete for 8 yards", 0.25),
        ("play", "Incomplete pass", 0.15),
        ("big_play", "Long pass for 30+ yards!", 0.05),
        ("touchdown", "TOUCHDOWN!", 0.03),
        ("turnover", "Interception!", 0.02),
        ("turnover", "Fumble lost!", 0.02),
        ("field_goal", "Field goal attempt", 0.04),
        ("punt", "Punt", 0.04),
    ]
    
    def __init__(self):
        self._games: Dict[str, RecordedGame] = {}
        self._active_simulations: Dict[str, bool] = {}
    
    def load_game(self, game_id: str, events: List[dict]) -> RecordedGame:
        """Load a game from event data"""
        game = RecordedGame(
            game_id=game_id,
            home_team=events[0].get("home_team", "HOME") if events else "HOME",
            away_team=events[0].get("away_team", "AWAY") if events else "AWAY",
            events=[
                GameEvent(
                    event_id=e.get("event_id", f"{game_id}_{i}"),
                    event_type=e.get("type", "play"),
                    description=e.get("description", ""),
                    team=e.get("team", ""),
                    timestamp=e.get("timestamp", i * 30),  # Default 30s between events
                    metadata=e.get("metadata", {})
                )
                for i, e in enumerate(events)
            ]
        )
        
        if game.events:
            game.duration_seconds = max(e.timestamp for e in game.events)
        
        self._games[game_id] = game
        return game
    
    def load_from_file(self, file_path: str) -> RecordedGame:
        """Load game from JSON file"""
        with open(file_path) as f:
            data = json.load(f)
        return self.load_game(data["game_id"], data["events"])
    
    async def replay(
        self, 
        game_id: str, 
        speed: float = 1.0,
        on_event: Optional[Callable] = None
    ) -> AsyncGenerator[GameEvent, None]:
        """
        Replay a game at specified speed.
        
        Args:
            game_id: Game to replay
            speed: Playback speed (1.0 = real-time, 10.0 = 10x faster)
            on_event: Optional callback for each event
        """
        if game_id not in self._games:
            raise ValueError(f"Game not found: {game_id}")
        
        game = self._games[game_id]
        self._active_simulations[game_id] = True
        
        start_time = time.time()
        last_event_time = 0
        
        for event in game.events:
            if not self._active_simulations.get(game_id):
                break
            
            # Calculate delay
            time_diff = event.timestamp - last_event_time
            delay = time_diff / speed
            
            if delay > 0:
                await asyncio.sleep(delay)
            
            last_event_time = event.timestamp
            
            if on_event:
                on_event(event)
            
            yield event
        
        self._active_simulations[game_id] = False
    
    async def generate_random(
        self,
        game_id: str = "random_game",
        duration_minutes: int = 60,
        events_per_minute: float = 2.0,
        speed: float = 60.0  # Default 60x speed
    ) -> AsyncGenerator[GameEvent, None]:
        """Generate random events for testing"""
        import random
        
        self._active_simulations[game_id] = True
        
        total_events = int(duration_minutes * events_per_minute)
        interval_seconds = (duration_minutes * 60) / total_events / speed
        
        teams = ["KC", "BUF"]
        
        for i in range(total_events):
            if not self._active_simulations.get(game_id):
                break
            
            # Pick event type based on weights
            rand = random.random()
            cumulative = 0
            event_type = "play"
            description = "Play"
            
            for etype, desc, weight in self.SAMPLE_EVENTS:
                cumulative += weight
                if rand <= cumulative:
                    event_type = etype
                    description = desc
                    break
            
            event = GameEvent(
                event_id=f"{game_id}_{i}",
                event_type=event_type,
                description=description,
                team=random.choice(teams),
                timestamp=i * (60 / events_per_minute)
            )
            
            await asyncio.sleep(interval_seconds)
            yield event
        
        self._active_simulations[game_id] = False
    
    def stop(self, game_id: str):
        """Stop a running simulation"""
        self._active_simulations[game_id] = False
    
    def stop_all(self):
        """Stop all simulations"""
        for game_id in self._active_simulations:
            self._active_simulations[game_id] = False
    
    def get_game(self, game_id: str) -> Optional[dict]:
        """Get game info"""
        if game_id not in self._games:
            return None
        
        game = self._games[game_id]
        return {
            "game_id": game.game_id,
            "home_team": game.home_team,
            "away_team": game.away_team,
            "event_count": len(game.events),
            "duration_seconds": game.duration_seconds,
            "is_active": self._active_simulations.get(game_id, False)
        }
    
    def list_games(self) -> List[str]:
        """List loaded games"""
        return list(self._games.keys())


# Sample game for testing
SAMPLE_GAME = {
    "game_id": "sample_kc_buf",
    "events": [
        {"type": "play", "description": "KC wins the toss, defers", "team": "KC", "timestamp": 0},
        {"type": "play", "description": "Kickoff to BUF, returned to 25", "team": "BUF", "timestamp": 10},
        {"type": "play", "description": "Allen pass to Diggs for 12 yards", "team": "BUF", "timestamp": 45},
        {"type": "play", "description": "Run up the middle for 3 yards", "team": "BUF", "timestamp": 90},
        {"type": "big_play", "description": "Allen to Davis for 35 yards!", "team": "BUF", "timestamp": 120},
        {"type": "touchdown", "description": "TOUCHDOWN BUF! Allen to Knox!", "team": "BUF", "timestamp": 180},
        {"type": "play", "description": "Kickoff to KC, touchback", "team": "KC", "timestamp": 210},
        {"type": "play", "description": "Mahomes scrambles for 8 yards", "team": "KC", "timestamp": 240},
        {"type": "big_play", "description": "Mahomes to Kelce for 40 yards!", "team": "KC", "timestamp": 280},
        {"type": "touchdown", "description": "TOUCHDOWN KC! Mahomes to Rice!", "team": "KC", "timestamp": 320},
    ]
}


# Singleton
_simulator: Optional[MockGameSimulator] = None

def get_game_simulator() -> MockGameSimulator:
    global _simulator
    if _simulator is None:
        _simulator = MockGameSimulator()
        _simulator.load_game(SAMPLE_GAME["game_id"], SAMPLE_GAME["events"])
    return _simulator
