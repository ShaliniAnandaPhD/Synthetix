"""
NFL Event Feed Connector

Connects to real-time NFL play-by-play data sources.
Supports multiple feeds: ESPN, Sportradar (if available), or mock data for testing.
"""

import asyncio
import json
import time
import logging
from typing import AsyncGenerator, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class NFLGame:
    """Active NFL game info"""
    game_id: str
    home_team: str
    away_team: str
    home_score: int = 0
    away_score: int = 0
    quarter: int = 1
    clock: str = "15:00"
    status: str = "scheduled"  # scheduled, in_progress, halftime, final


@dataclass 
class NFLPlay:
    """A single NFL play event"""
    game_id: str
    play_id: str
    description: str
    type: str  # pass, rush, kick, penalty, timeout, etc.
    team: str
    players: list = field(default_factory=list)
    yards: int = 0
    down: int = 0
    distance: int = 0
    result: str = ""  # touchdown, first_down, incomplete, etc.
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "play_id": self.play_id,
            "description": self.description,
            "type": self.type,
            "team": self.team,
            "players": self.players,
            "yards": self.yards,
            "down": self.down,
            "distance": self.distance,
            "result": self.result,
            "timestamp": self.timestamp
        }


class NFLEventFeed:
    """
    Real-time NFL event feed connector.
    
    Supports:
    - ESPN unofficial API (free, may break)
    - Mock data for testing
    - Custom webhook sources
    
    Usage:
        feed = NFLEventFeed()
        async for play in feed.stream_game("401547594"):
            print(play.description)
    """
    
    ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
    
    def __init__(
        self,
        source: str = "espn",  # espn, mock, or webhook
        poll_interval: float = 2.0,
        api_key: Optional[str] = None
    ):
        """
        Initialize the event feed.
        
        Args:
            source: Data source - "espn", "mock", or "webhook"
            poll_interval: Seconds between polls (for polling sources)
            api_key: Optional API key for premium sources
        """
        self.source = source
        self.poll_interval = poll_interval
        self.api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._last_play_ids: Dict[str, set] = {}  # Track seen plays per game
    
    async def start(self):
        """Start the feed connection"""
        self._session = aiohttp.ClientSession()
        self._running = True
        logger.info(f"NFLEventFeed started with source: {self.source}")
    
    async def stop(self):
        """Stop the feed connection"""
        self._running = False
        if self._session:
            await self._session.close()
            self._session = None
        logger.info("NFLEventFeed stopped")
    
    async def get_live_games(self) -> list[NFLGame]:
        """Get all currently live NFL games"""
        if self.source == "mock":
            return self._get_mock_games()
        
        try:
            url = f"{self.ESPN_BASE}/scoreboard"
            async with self._session.get(url) as resp:
                if resp.status != 200:
                    logger.warning(f"ESPN API returned {resp.status}")
                    return []
                
                data = await resp.json()
                games = []
                
                for event in data.get("events", []):
                    competition = event.get("competitions", [{}])[0]
                    competitors = competition.get("competitors", [])
                    
                    if len(competitors) < 2:
                        continue
                    
                    home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
                    away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])
                    
                    status = competition.get("status", {})
                    
                    games.append(NFLGame(
                        game_id=event.get("id"),
                        home_team=home.get("team", {}).get("abbreviation", "UNK"),
                        away_team=away.get("team", {}).get("abbreviation", "UNK"),
                        home_score=int(home.get("score", 0)),
                        away_score=int(away.get("score", 0)),
                        quarter=status.get("period", 1),
                        clock=status.get("displayClock", "15:00"),
                        status=status.get("type", {}).get("name", "scheduled")
                    ))
                
                return games
                
        except Exception as e:
            logger.error(f"Error fetching live games: {e}")
            return []
    
    async def stream_game(
        self, 
        game_id: str,
        on_play: Optional[Callable[[NFLPlay], None]] = None
    ) -> AsyncGenerator[NFLPlay, None]:
        """
        Stream plays from a specific game.
        
        Args:
            game_id: ESPN game ID
            on_play: Optional callback for each play
        
        Yields:
            NFLPlay objects as they occur
        """
        if game_id not in self._last_play_ids:
            self._last_play_ids[game_id] = set()
        
        while self._running:
            try:
                if self.source == "mock":
                    async for play in self._mock_game_stream(game_id):
                        if on_play:
                            on_play(play)
                        yield play
                else:
                    plays = await self._fetch_game_plays(game_id)
                    
                    for play in plays:
                        if play.play_id not in self._last_play_ids[game_id]:
                            self._last_play_ids[game_id].add(play.play_id)
                            if on_play:
                                on_play(play)
                            yield play
                
                await asyncio.sleep(self.poll_interval)
                
            except Exception as e:
                logger.error(f"Error streaming game {game_id}: {e}")
                await asyncio.sleep(5)  # Back off on error
    
    async def _fetch_game_plays(self, game_id: str) -> list[NFLPlay]:
        """Fetch plays for a specific game from ESPN"""
        try:
            url = f"{self.ESPN_BASE}/summary?event={game_id}"
            async with self._session.get(url) as resp:
                if resp.status != 200:
                    return []
                
                data = await resp.json()
                plays = []
                
                # Parse drives and plays
                for drive in data.get("drives", {}).get("plays", []):
                    plays.append(self._parse_espn_play(game_id, drive))
                
                return plays
                
        except Exception as e:
            logger.error(f"Error fetching plays for {game_id}: {e}")
            return []
    
    def _parse_espn_play(self, game_id: str, play_data: dict) -> NFLPlay:
        """Parse ESPN play data into NFLPlay"""
        return NFLPlay(
            game_id=game_id,
            play_id=str(play_data.get("id", time.time())),
            description=play_data.get("text", ""),
            type=play_data.get("type", {}).get("text", "play"),
            team=play_data.get("team", {}).get("abbreviation", ""),
            yards=play_data.get("yards", 0),
            down=play_data.get("start", {}).get("down", 0),
            distance=play_data.get("start", {}).get("distance", 0),
            result=play_data.get("result", {}).get("type", ""),
            timestamp=time.time()
        )
    
    def _get_mock_games(self) -> list[NFLGame]:
        """Return mock games for testing"""
        return [
            NFLGame(
                game_id="mock_kc_buf",
                home_team="BUF",
                away_team="KC",
                home_score=21,
                away_score=24,
                quarter=4,
                clock="2:00",
                status="in_progress"
            ),
            NFLGame(
                game_id="mock_dal_phi",
                home_team="PHI",
                away_team="DAL",
                home_score=17,
                away_score=14,
                quarter=3,
                clock="8:32",
                status="in_progress"
            )
        ]
    
    async def _mock_game_stream(self, game_id: str) -> AsyncGenerator[NFLPlay, None]:
        """Generate mock plays for testing"""
        mock_plays = [
            NFLPlay(
                game_id=game_id,
                play_id=f"mock_{int(time.time())}",
                description="Patrick Mahomes pass complete to Travis Kelce for 25 yards. First down!",
                type="pass",
                team="KC",
                players=["Patrick Mahomes", "Travis Kelce"],
                yards=25,
                result="first_down"
            ),
            NFLPlay(
                game_id=game_id,
                play_id=f"mock_{int(time.time()) + 1}",
                description="Josh Allen TOUCHDOWN pass to Stefon Diggs! 12 yards!",
                type="pass",
                team="BUF",
                players=["Josh Allen", "Stefon Diggs"],
                yards=12,
                result="touchdown"
            ),
            NFLPlay(
                game_id=game_id,
                play_id=f"mock_{int(time.time()) + 2}",
                description="Isiah Pacheco rush up the middle for 8 yards",
                type="rush",
                team="KC",
                players=["Isiah Pacheco"],
                yards=8,
                result="first_down"
            ),
        ]
        
        for play in mock_plays:
            await asyncio.sleep(3)  # Simulate real-time delay
            yield play


# Factory function
def create_event_feed(source: str = "espn", **kwargs) -> NFLEventFeed:
    """Create an NFL event feed instance"""
    return NFLEventFeed(source=source, **kwargs)


if __name__ == "__main__":
    async def test():
        feed = NFLEventFeed(source="mock")
        await feed.start()
        
        print("Getting live games...")
        games = await feed.get_live_games()
        for game in games:
            print(f"  {game.away_team} @ {game.home_team}: {game.away_score}-{game.home_score} (Q{game.quarter})")
        
        if games:
            print(f"\nStreaming plays from {games[0].game_id}...")
            count = 0
            async for play in feed.stream_game(games[0].game_id):
                print(f"  {play.type.upper()}: {play.description}")
                count += 1
                if count >= 3:
                    break
        
        await feed.stop()
    
    asyncio.run(test())
