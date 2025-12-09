"""
Event Classifier for Live Commentary

Categorizes incoming NFL events by urgency and commentary type.
This is the foundation of the live commentary pipeline.

Urgency Levels:
- IMMEDIATE: React in <150ms (touchdowns, turnovers, injuries)
- STRATEGIC: Can analyze in <500ms (drive stalls, timeouts)
- CONTEXTUAL: Fill time thoughtfully <2000ms (stats, historical notes)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
import time
import re


class EventUrgency(Enum):
    """How quickly we need to respond to this event"""
    IMMEDIATE = "immediate"    # Touchdown, turnover, injury - react NOW
    STRATEGIC = "strategic"    # Drive stalls, timeout - can analyze
    CONTEXTUAL = "contextual"  # Stat milestone, historical note - fill time


class EventType(Enum):
    """Types of NFL events we can classify"""
    TOUCHDOWN = "touchdown"
    TURNOVER = "turnover"
    INTERCEPTION = "interception"
    FUMBLE = "fumble"
    BIG_PLAY = "big_play"           # 20+ yard gain
    INJURY = "injury"
    PENALTY = "penalty"
    SCORE_CHANGE = "score_change"
    FIELD_GOAL = "field_goal"
    CLOCK_STOP = "clock_stop"       # Timeout, 2-minute warning
    DRIVE_END = "drive_end"
    QUARTER_END = "quarter_end"
    HALFTIME = "halftime"
    GAME_START = "game_start"
    GAME_END = "game_end"
    INACTIVE_ANNOUNCEMENT = "inactive"
    WEATHER_UPDATE = "weather"
    STAT_MILESTONE = "stat_milestone"
    UNKNOWN = "unknown"


@dataclass
class ClassifiedEvent:
    """A classified NFL event with all metadata for commentary dispatch"""
    raw_event: dict
    event_type: EventType
    urgency: EventUrgency
    teams_involved: list[str] = field(default_factory=list)
    players_involved: list[str] = field(default_factory=list)
    fantasy_impact: float = 0.0         # 0-1 scale
    controversy_potential: float = 0.0  # 0-1 scale for triggering debate
    timestamp: float = field(default_factory=time.time)
    game_id: Optional[str] = None
    description: str = ""
    
    @property
    def latency_budget_ms(self) -> int:
        """How long we have to respond based on urgency"""
        if self.urgency == EventUrgency.IMMEDIATE:
            return 150  # React FAST
        elif self.urgency == EventUrgency.STRATEGIC:
            return 500  # Can think
        else:
            return 2000  # Fill time thoughtfully
    
    @property
    def should_interrupt(self) -> bool:
        """Should this event interrupt current commentary?"""
        return self.urgency == EventUrgency.IMMEDIATE
    
    def to_dict(self) -> dict:
        """Serialize for transmission"""
        return {
            "event_type": self.event_type.value,
            "urgency": self.urgency.value,
            "teams": self.teams_involved,
            "players": self.players_involved,
            "fantasy_impact": self.fantasy_impact,
            "controversy": self.controversy_potential,
            "latency_budget_ms": self.latency_budget_ms,
            "timestamp": self.timestamp,
            "game_id": self.game_id,
            "description": self.description
        }


class EventClassifier:
    """
    Classifies NFL events for live commentary routing.
    
    Usage:
        classifier = EventClassifier()
        event = classifier.classify(raw_nfl_event)
        print(f"Event: {event.event_type}, Urgency: {event.urgency}")
        print(f"Must respond within {event.latency_budget_ms}ms")
    """
    
    # Keywords for event type detection
    EVENT_KEYWORDS = {
        EventType.TOUCHDOWN: ["touchdown", "td", "scores", "scored", "end zone"],
        EventType.INTERCEPTION: ["interception", "picked off", "intercepted", "int"],
        EventType.FUMBLE: ["fumble", "fumbled", "strips", "stripped"],
        EventType.TURNOVER: ["turnover", "turnover on downs"],
        EventType.BIG_PLAY: ["big gain", "breaks free", "wide open"],
        EventType.INJURY: ["injury", "injured", "hurt", "down on the field", "medical"],
        EventType.PENALTY: ["penalty", "flag", "flagged", "illegal"],
        EventType.FIELD_GOAL: ["field goal", "fg", "kicks"],
        EventType.CLOCK_STOP: ["timeout", "two-minute warning", "clock stopped"],
        EventType.DRIVE_END: ["punt", "punts", "turnover on downs"],
        EventType.QUARTER_END: ["end of quarter", "quarter ends", "end of the"],
        EventType.HALFTIME: ["halftime", "half time"],
        EventType.STAT_MILESTONE: ["career high", "record", "milestone", "100 yards", "1000 yards"],
    }
    
    # Fantasy impact weights by event type
    FANTASY_IMPACT = {
        EventType.TOUCHDOWN: 1.0,
        EventType.INTERCEPTION: 0.8,
        EventType.FUMBLE: 0.7,
        EventType.BIG_PLAY: 0.6,
        EventType.FIELD_GOAL: 0.5,
        EventType.INJURY: 0.9,  # High impact on lineup decisions
        EventType.STAT_MILESTONE: 0.4,
    }
    
    # Controversy potential by event type (triggers debate)
    CONTROVERSY_POTENTIAL = {
        EventType.PENALTY: 0.8,
        EventType.TURNOVER: 0.6,
        EventType.INTERCEPTION: 0.7,
        EventType.INJURY: 0.5,
        EventType.TOUCHDOWN: 0.3,  # Celebration can be controversial
    }
    
    def __init__(self):
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for faster matching"""
        self._patterns = {}
        for event_type, keywords in self.EVENT_KEYWORDS.items():
            pattern = '|'.join(re.escape(kw) for kw in keywords)
            self._patterns[event_type] = re.compile(pattern, re.IGNORECASE)
    
    def classify(self, raw_event: dict) -> ClassifiedEvent:
        """
        Classify a raw NFL event.
        
        Args:
            raw_event: Dict with keys like 'description', 'type', 'teams', 'players', etc.
        
        Returns:
            ClassifiedEvent with urgency, type, and metadata
        """
        event_type = self._detect_type(raw_event)
        urgency = self._assess_urgency(event_type, raw_event)
        
        return ClassifiedEvent(
            raw_event=raw_event,
            event_type=event_type,
            urgency=urgency,
            teams_involved=self._extract_teams(raw_event),
            players_involved=self._extract_players(raw_event),
            fantasy_impact=self._calculate_fantasy_impact(event_type, raw_event),
            controversy_potential=self._assess_controversy(event_type, raw_event),
            timestamp=time.time(),
            game_id=raw_event.get("game_id"),
            description=raw_event.get("description", str(raw_event))
        )
    
    def _detect_type(self, raw_event: dict) -> EventType:
        """Detect event type from raw event data"""
        # Check explicit type field first
        if "type" in raw_event:
            type_str = raw_event["type"].lower()
            for event_type in EventType:
                if event_type.value in type_str:
                    return event_type
        
        # Check description for keywords
        description = raw_event.get("description", "")
        if isinstance(description, str):
            for event_type, pattern in self._patterns.items():
                if pattern.search(description):
                    return event_type
        
        # Check for yardage (big play detection)
        yards = raw_event.get("yards", 0)
        if isinstance(yards, (int, float)) and yards >= 20:
            return EventType.BIG_PLAY
        
        return EventType.UNKNOWN
    
    def _assess_urgency(self, event_type: EventType, raw_event: dict) -> EventUrgency:
        """Determine how urgently we need to respond"""
        immediate_events = {
            EventType.TOUCHDOWN,
            EventType.INTERCEPTION,
            EventType.FUMBLE,
            EventType.TURNOVER,
            EventType.BIG_PLAY,
            EventType.INJURY,
        }
        
        strategic_events = {
            EventType.DRIVE_END,
            EventType.PENALTY,
            EventType.FIELD_GOAL,
            EventType.CLOCK_STOP,
        }
        
        if event_type in immediate_events:
            return EventUrgency.IMMEDIATE
        elif event_type in strategic_events:
            return EventUrgency.STRATEGIC
        return EventUrgency.CONTEXTUAL
    
    def _extract_teams(self, raw_event: dict) -> list[str]:
        """Extract team names from event"""
        teams = []
        
        # Check common fields
        for field in ["team", "teams", "home_team", "away_team", "offense", "defense"]:
            if field in raw_event:
                value = raw_event[field]
                if isinstance(value, str):
                    teams.append(value)
                elif isinstance(value, list):
                    teams.extend(value)
        
        return list(set(teams))
    
    def _extract_players(self, raw_event: dict) -> list[str]:
        """Extract player names from event"""
        players = []
        
        # Check common fields
        for field in ["player", "players", "passer", "rusher", "receiver", "tackler"]:
            if field in raw_event:
                value = raw_event[field]
                if isinstance(value, str):
                    players.append(value)
                elif isinstance(value, list):
                    players.extend(value)
        
        # Parse description for player names (basic pattern: First Last)
        description = raw_event.get("description", "")
        if isinstance(description, str):
            # Match "FirstName LastName" patterns
            name_pattern = r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b'
            matches = re.findall(name_pattern, description)
            players.extend(matches)
        
        return list(set(players))[:5]  # Limit to 5 players
    
    def _calculate_fantasy_impact(self, event_type: EventType, raw_event: dict) -> float:
        """Calculate fantasy football impact of this event"""
        base_impact = self.FANTASY_IMPACT.get(event_type, 0.3)
        
        # Boost for high-value plays
        yards = raw_event.get("yards", 0)
        if isinstance(yards, (int, float)):
            if yards >= 50:
                base_impact = min(1.0, base_impact + 0.2)
            elif yards >= 30:
                base_impact = min(1.0, base_impact + 0.1)
        
        # Boost for red zone plays
        if raw_event.get("red_zone", False):
            base_impact = min(1.0, base_impact + 0.15)
        
        return round(base_impact, 2)
    
    def _assess_controversy(self, event_type: EventType, raw_event: dict) -> float:
        """Assess how controversial/debatable this event is"""
        base_controversy = self.CONTROVERSY_POTENTIAL.get(event_type, 0.2)
        
        # Boost for close calls
        if raw_event.get("reviewed", False) or raw_event.get("challenged", False):
            base_controversy = min(1.0, base_controversy + 0.3)
        
        # Boost for rivalry games
        if raw_event.get("rivalry", False):
            base_controversy = min(1.0, base_controversy + 0.2)
        
        return round(base_controversy, 2)


# Convenience function for quick classification
def classify_event(raw_event: dict) -> ClassifiedEvent:
    """Quick classification of a single event"""
    classifier = EventClassifier()
    return classifier.classify(raw_event)


if __name__ == "__main__":
    # Test classification
    test_events = [
        {"description": "Patrick Mahomes pass complete to Travis Kelce for a TOUCHDOWN!", "type": "play"},
        {"description": "Josh Allen INTERCEPTED by Sauce Gardner!", "yards": 0},
        {"description": "Derrick Henry breaks free for a 45-yard gain!", "yards": 45},
        {"description": "Injury timeout on the field. Medical staff attending to player.", "type": "timeout"},
        {"description": "Pass incomplete. Third down.", "type": "play"},
    ]
    
    classifier = EventClassifier()
    for event in test_events:
        result = classifier.classify(event)
        print(f"\nEvent: {event.get('description', event)[:50]}...")
        print(f"  Type: {result.event_type.value}")
        print(f"  Urgency: {result.urgency.value}")
        print(f"  Latency Budget: {result.latency_budget_ms}ms")
        print(f"  Fantasy Impact: {result.fantasy_impact}")
        print(f"  Controversy: {result.controversy_potential}")
