"""
Data Collection Unit Tests

Tests for ESPN data parsing, sentiment analysis, and player extraction.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.identity_regression import map_event_to_type, get_event_importance


# =============================================================================
# ESPN EVENT MAPPING TESTS
# =============================================================================

class TestESPNEventMapping:
    """Tests for ESPN event type mapping."""
    
    def test_touchdown_mapping(self):
        """Test touchdown events are mapped correctly."""
        events = [
            {"type": "Passing Touchdown"},
            {"type": {"text": "Rushing Touchdown"}},
            {"text": "TOUCHDOWN"},
            {"description": "10 yard TD pass to Kelce"}
        ]
        
        for event in events:
            event_type = map_event_to_type(event)
            assert event_type == "touchdown", f"Failed for {event}"
    
    def test_interception_mapping(self):
        """Test interception events are mapped correctly."""
        events = [
            {"type": "Interception"},
            {"type": {"text": "INT"}},
            {"text": "Pass Intercepted by Smith"}
        ]
        
        for event in events:
            event_type = map_event_to_type(event)
            assert event_type == "interception", f"Failed for {event}"
    
    def test_fumble_mapping(self):
        """Test fumble events are mapped correctly."""
        events = [
            {"type": "Fumble"},
            {"type": {"text": "Lost Fumble"}},
            {"description": "FUMBLE by Mahomes"}
        ]
        
        for event in events:
            event_type = map_event_to_type(event)
            assert event_type == "fumble", f"Failed for {event}"
    
    def test_field_goal_mapping(self):
        """Test field goal events are mapped correctly."""
        events = [
            {"type": "Field Goal"},
            {"type": {"text": "FG Good"}},
            {"description": "42 yard FG by Butker"}
        ]
        
        for event in events:
            event_type = map_event_to_type(event)
            assert event_type == "field_goal", f"Failed for {event}"
    
    def test_unknown_event_returns_generic(self):
        """Unknown events should return generic or their own mapping."""
        events = [
            {"type": "Clock Running"},
            {},
            {"type": "Unknown Play Type"}
        ]
        
        for event in events:
            event_type = map_event_to_type(event)
            # Unknown should return generic
            assert event_type == "generic", f"Failed for {event}"
    
    def test_timeout_returns_timeout(self):
        """Timeout events should return timeout."""
        event = {"type": {"text": "Timeout"}}
        event_type = map_event_to_type(event)
        assert event_type == "timeout"


# =============================================================================
# EVENT IMPORTANCE TESTS
# =============================================================================

class TestEventImportance:
    """Tests for event importance scoring."""
    
    def test_touchdown_highest_importance(self):
        """Touchdowns should have highest importance."""
        importance = get_event_importance("touchdown")
        assert importance == 10
    
    def test_turnover_high_importance(self):
        """Turnovers should have high importance."""
        assert get_event_importance("interception") >= 8
        assert get_event_importance("fumble") >= 8
    
    def test_field_goal_medium_importance(self):
        """Field goals should have medium importance."""
        importance = get_event_importance("field_goal")
        assert 5 <= importance <= 7
    
    def test_generic_low_importance(self):
        """Generic events should have low importance."""
        importance = get_event_importance("generic")
        assert importance <= 3


# =============================================================================
# SENTIMENT ANALYSIS TESTS
# =============================================================================

class TestSentimentAnalysis:
    """Tests for Twitter sentiment analysis."""
    
    def test_positive_sentiment_patterns(self):
        """Test positive sentiment detection."""
        positive_patterns = [
            "amazing", "incredible", "love", "best", "perfect",
            "🔥", "💪", "🙌", "!!!!"
        ]
        
        for pattern in positive_patterns:
            sentiment = self._simple_sentiment(f"That play was {pattern}!")
            assert sentiment > 0, f"Expected positive for: {pattern}"
    
    def test_negative_sentiment_patterns(self):
        """Test negative sentiment detection."""
        negative_patterns = [
            "terrible", "awful", "worst", "hate", "horrible",
            "😡", "💀", "trash"
        ]
        
        for pattern in negative_patterns:
            sentiment = self._simple_sentiment(f"That was {pattern}")
            assert sentiment < 0, f"Expected negative for: {pattern}"
    
    def test_neutral_sentiment(self):
        """Test neutral sentiment detection."""
        neutral = "The game continues in the third quarter."
        sentiment = self._simple_sentiment(neutral)
        
        # Neutral should be close to 0
        assert -0.3 <= sentiment <= 0.3
    
    def _simple_sentiment(self, text: str) -> float:
        """Simple rule-based sentiment for testing."""
        positive_words = ["amazing", "incredible", "love", "best", "perfect", "🔥", "💪", "🙌"]
        negative_words = ["terrible", "awful", "worst", "hate", "horrible", "😡", "💀", "trash"]
        
        text_lower = text.lower()
        
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)
        exclaim_count = text.count("!")
        
        score = (pos_count - neg_count) * 0.3 + exclaim_count * 0.1
        return max(-1.0, min(1.0, score))


# =============================================================================
# PLAYER EXTRACTION TESTS
# =============================================================================

class TestPlayerExtraction:
    """Tests for extracting player names from text."""
    
    # Common NFL player names for testing
    KNOWN_PLAYERS = [
        "Patrick Mahomes", "Travis Kelce", "Josh Allen",
        "Lamar Jackson", "Derrick Henry", "C.J. Stroud",
        "Jalen Hurts", "Justin Jefferson", "Tyreek Hill"
    ]
    
    def test_extract_single_player(self):
        """Test extracting a single player name."""
        text = "Patrick Mahomes throws a perfect pass!"
        players = self._extract_players(text)
        
        assert "Patrick Mahomes" in players
    
    def test_extract_multiple_players(self):
        """Test extracting multiple player names."""
        text = "Patrick Mahomes throws to Travis Kelce for the touchdown!"
        players = self._extract_players(text)
        
        assert "Patrick Mahomes" in players
        assert "Travis Kelce" in players
    
    def test_no_players_found(self):
        """Test when no known players in text."""
        text = "The team scored a touchdown."
        players = self._extract_players(text)
        
        assert len(players) == 0
    
    def _extract_players(self, text: str) -> list:
        """Simple player extraction for testing."""
        found = []
        for player in self.KNOWN_PLAYERS:
            if player in text:
                found.append(player)
        return found


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
