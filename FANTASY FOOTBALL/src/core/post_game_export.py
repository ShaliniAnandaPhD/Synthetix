"""
Post-Game Export

Converts live commentary session into downloadable debate format.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class CommentaryEntry:
    """A single commentary entry from live session"""
    timestamp: float
    region: str
    agent_type: str
    text: str
    emotion: str
    event_type: Optional[str] = None
    audio_url: Optional[str] = None


@dataclass
class GameSession:
    """Complete live session data"""
    game_id: str
    home_team: str
    away_team: str
    started_at: float
    ended_at: Optional[float] = None
    final_score: Optional[str] = None
    entries: List[CommentaryEntry] = field(default_factory=list)
    regions: List[str] = field(default_factory=list)


class PostGameExporter:
    """
    Exports live commentary sessions to various formats.
    
    Features:
    - Convert to debate transcript format
    - Export as SRT subtitles
    - Generate highlight clips metadata
    - Save to Supabase for replay
    
    Usage:
        exporter = PostGameExporter()
        
        # During game, collect entries
        exporter.add_entry("KC_BUF", entry)
        
        # After game
        transcript = exporter.export_transcript("KC_BUF")
        srt = exporter.export_srt("KC_BUF")
    """
    
    def __init__(self, storage_client=None):
        self.storage = storage_client
        self.sessions: Dict[str, GameSession] = {}
    
    def start_session(
        self, 
        game_id: str, 
        home_team: str, 
        away_team: str,
        regions: List[str]
    ) -> GameSession:
        """Start a new session for recording"""
        session = GameSession(
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
            started_at=time.time(),
            regions=regions
        )
        self.sessions[game_id] = session
        return session
    
    def add_entry(
        self,
        game_id: str,
        region: str,
        agent_type: str,
        text: str,
        emotion: str = "neutral",
        event_type: Optional[str] = None,
        audio_url: Optional[str] = None
    ):
        """Add a commentary entry to the session"""
        if game_id not in self.sessions:
            return
        
        entry = CommentaryEntry(
            timestamp=time.time(),
            region=region,
            agent_type=agent_type,
            text=text,
            emotion=emotion,
            event_type=event_type,
            audio_url=audio_url
        )
        self.sessions[game_id].entries.append(entry)
    
    def end_session(self, game_id: str, final_score: Optional[str] = None):
        """End a recording session"""
        if game_id not in self.sessions:
            return
        
        session = self.sessions[game_id]
        session.ended_at = time.time()
        session.final_score = final_score
    
    def export_transcript(self, game_id: str) -> Optional[str]:
        """
        Export as readable transcript.
        
        Format:
        [00:00:00] 🏈 KANSAS_CITY (homer): TOUCHDOWN! That's what I'm talking about!
        [00:00:02] 📊 KANSAS_CITY (analyst): Perfect execution on that drive.
        """
        if game_id not in self.sessions:
            return None
        
        session = self.sessions[game_id]
        lines = []
        
        # Header
        lines.append(f"# {session.away_team} @ {session.home_team}")
        lines.append(f"Date: {datetime.fromtimestamp(session.started_at).isoformat()}")
        if session.final_score:
            lines.append(f"Final Score: {session.final_score}")
        lines.append(f"Regions: {', '.join(session.regions)}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Entries
        start_time = session.started_at
        
        for entry in session.entries:
            elapsed = entry.timestamp - start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            
            timestamp = f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"
            
            emoji = self._get_agent_emoji(entry.agent_type)
            region = entry.region.upper()
            
            lines.append(f"{timestamp} {emoji} {region} ({entry.agent_type}): {entry.text}")
        
        return "\n".join(lines)
    
    def export_srt(self, game_id: str) -> Optional[str]:
        """Export as SRT subtitle format"""
        if game_id not in self.sessions:
            return None
        
        session = self.sessions[game_id]
        srt_lines = []
        start_time = session.started_at
        
        for i, entry in enumerate(session.entries, 1):
            # Calculate times
            start = entry.timestamp - start_time
            # Estimate duration based on text length (~150 chars/minute)
            duration = max(2, len(entry.text) / 15)
            end = start + duration
            
            # Format timestamps
            start_str = self._format_srt_time(start)
            end_str = self._format_srt_time(end)
            
            srt_lines.append(str(i))
            srt_lines.append(f"{start_str} --> {end_str}")
            srt_lines.append(f"[{entry.region.upper()}] {entry.text}")
            srt_lines.append("")
        
        return "\n".join(srt_lines)
    
    def export_debate_format(self, game_id: str) -> Optional[dict]:
        """
        Export in the format used by DebatePlaybackEngine.
        
        Can be loaded into the existing debate replay system.
        """
        if game_id not in self.sessions:
            return None
        
        session = self.sessions[game_id]
        start_time = session.started_at
        
        segments = []
        for entry in session.entries:
            segments.append({
                "speaker_id": f"{entry.region}_{entry.agent_type}",
                "speaker_name": f"{entry.region.replace('_', ' ').title()} {entry.agent_type.title()}",
                "text": entry.text,
                "start_time": entry.timestamp - start_time,
                "audio_url": entry.audio_url,
                "emotion": entry.emotion
            })
        
        return {
            "id": game_id,
            "title": f"{session.away_team} @ {session.home_team} Live Commentary",
            "created_at": datetime.fromtimestamp(session.started_at).isoformat(),
            "duration": (session.ended_at or time.time()) - session.started_at,
            "type": "live_commentary",
            "segments": segments,
            "metadata": {
                "game_id": game_id,
                "home_team": session.home_team,
                "away_team": session.away_team,
                "final_score": session.final_score,
                "regions": session.regions
            }
        }
    
    def export_highlights(
        self, 
        game_id: str,
        event_types: List[str] = None
    ) -> Optional[List[dict]]:
        """
        Export only highlight moments (touchdowns, turnovers, etc.)
        """
        if game_id not in self.sessions:
            return None
        
        session = self.sessions[game_id]
        
        if event_types is None:
            event_types = ["touchdown", "turnover", "interception", "big_play"]
        
        highlights = []
        start_time = session.started_at
        
        for entry in session.entries:
            if entry.event_type and entry.event_type in event_types:
                highlights.append({
                    "timestamp": entry.timestamp - start_time,
                    "event_type": entry.event_type,
                    "region": entry.region,
                    "agent_type": entry.agent_type,
                    "text": entry.text,
                    "audio_url": entry.audio_url
                })
        
        return highlights
    
    async def save_to_storage(self, game_id: str) -> bool:
        """Save session to Supabase/storage"""
        if not self.storage or game_id not in self.sessions:
            return False
        
        try:
            debate_format = self.export_debate_format(game_id)
            
            # Save to storage (Supabase)
            await self.storage.save(f"live_sessions/{game_id}.json", debate_format)
            
            logger.info(f"Saved session {game_id} to storage")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            return False
    
    def _format_srt_time(self, seconds: float) -> str:
        """Format seconds as SRT timestamp (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"
    
    def _get_agent_emoji(self, agent_type: str) -> str:
        """Get emoji for agent type"""
        emojis = {
            "homer": "🏈",
            "analyst": "📊",
            "contrarian": "🤔",
            "historian": "📚",
            "stats_expert": "🔢",
        }
        return emojis.get(agent_type, "🎙️")
    
    def get_session_stats(self, game_id: str) -> Optional[dict]:
        """Get stats for a session"""
        if game_id not in self.sessions:
            return None
        
        session = self.sessions[game_id]
        
        entries_by_region = {}
        entries_by_agent = {}
        
        for entry in session.entries:
            entries_by_region[entry.region] = entries_by_region.get(entry.region, 0) + 1
            entries_by_agent[entry.agent_type] = entries_by_agent.get(entry.agent_type, 0) + 1
        
        return {
            "game_id": game_id,
            "total_entries": len(session.entries),
            "duration_seconds": (session.ended_at or time.time()) - session.started_at,
            "entries_by_region": entries_by_region,
            "entries_by_agent": entries_by_agent,
            "is_complete": session.ended_at is not None
        }


# Singleton
_exporter: Optional[PostGameExporter] = None

def get_exporter() -> PostGameExporter:
    global _exporter
    if _exporter is None:
        _exporter = PostGameExporter()
    return _exporter
