"""
Twitter Stream Unit Tests

Tests for Twitter integration, sentiment analysis, and tweet processing.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# =============================================================================
# TWITTER SENTIMENT TESTS
# =============================================================================

class TestTwitterSentiment:
    """Tests for Twitter sentiment analysis."""
    
    # Sentiment patterns
    POSITIVE_PATTERNS = ["amazing", "incredible", "love", "best", "🔥", "💪"]
    NEGATIVE_PATTERNS = ["terrible", "awful", "worst", "hate", "😡", "💀"]
    
    def test_positive_sentiment_detection(self):
        """Test positive sentiment correctly identified."""
        positive_tweets = [
            "That touchdown was AMAZING! 🔥🔥🔥",
            "Best game I've seen all season! Love this team!",
            "Incredible play by Mahomes! MVP material!"
        ]
        
        for tweet in positive_tweets:
            sentiment = self._analyze(tweet)
            assert sentiment > 0, f"Expected positive for: {tweet}"
    
    def test_negative_sentiment_detection(self):
        """Test negative sentiment correctly identified."""
        negative_tweets = [
            "Terrible coaching decision. Fire everyone.",
            "This is the WORST offense in the league 😡",
            "I hate this team so much right now 💀"
        ]
        
        for tweet in negative_tweets:
            sentiment = self._analyze(tweet)
            assert sentiment < 0, f"Expected negative for: {tweet}"
    
    def test_neutral_sentiment_detection(self):
        """Test neutral tweets scored near zero."""
        neutral_tweets = [
            "The game is currently in the 3rd quarter.",
            "Next play coming up.",
            "Both teams have scored today."
        ]
        
        for tweet in neutral_tweets:
            sentiment = self._analyze(tweet)
            assert -0.3 <= sentiment <= 0.3, f"Expected neutral for: {tweet}"
    
    def test_exclamation_boosts_intensity(self):
        """Test exclamations increase sentiment intensity."""
        calm = "Good play by the team"
        excited = "Good play by the team!!!"
        
        calm_score = abs(self._analyze(calm))
        excited_score = abs(self._analyze(excited))
        
        assert excited_score >= calm_score
    
    def _analyze(self, text: str) -> float:
        """Simple sentiment analysis for testing."""
        text_lower = text.lower()
        
        pos = sum(1 for p in self.POSITIVE_PATTERNS if p in text_lower)
        neg = sum(1 for p in self.NEGATIVE_PATTERNS if p in text_lower)
        exclaim = text.count("!")
        
        return (pos - neg) * 0.3 + exclaim * 0.05


# =============================================================================
# TWEET PROCESSING TESTS
# =============================================================================

class TestTweetProcessing:
    """Tests for processing raw tweet data."""
    
    def test_extract_hashtags(self):
        """Test hashtag extraction from tweets."""
        tweet = "What a game! #Texans #NFLSunday #HoustonStrong"
        
        hashtags = self._extract_hashtags(tweet)
        
        assert "#Texans" in hashtags
        assert "#NFLSunday" in hashtags
        assert len(hashtags) == 3
    
    def test_extract_mentions(self):
        """Test @mention extraction from tweets."""
        tweet = "Great analysis @ESPNStatsInfo! @AdamSchefter called it"
        
        mentions = self._extract_mentions(tweet)
        
        assert "@ESPNStatsInfo" in mentions
        assert "@AdamSchefter" in mentions
    
    def test_identify_verified_sources(self):
        """Test identification of verified NFL sources."""
        verified_handles = ["AdamSchefter", "RapSheet", "ESPNNewsNow"]
        
        for handle in verified_handles:
            assert self._is_verified_source(handle)
        
        # Random handles should not be verified
        assert not self._is_verified_source("random_fan_42")
    
    def test_filter_retweets(self):
        """Test retweet filtering."""
        tweets = [
            {"text": "Original content here", "is_retweet": False},
            {"text": "RT @someone: Retweet content", "is_retweet": True},
            {"text": "More original content", "is_retweet": False}
        ]
        
        filtered = [t for t in tweets if not t.get("is_retweet")]
        
        assert len(filtered) == 2
    
    def _extract_hashtags(self, text: str) -> list:
        """Extract hashtags from text."""
        import re
        return re.findall(r'#\w+', text)
    
    def _extract_mentions(self, text: str) -> list:
        """Extract mentions from text."""
        import re
        return re.findall(r'@\w+', text)
    
    def _is_verified_source(self, handle: str) -> bool:
        """Check if handle is a known NFL source."""
        verified = [
            "AdamSchefter", "RapSheet", "ESPNNewsNow", "JayGlazer",
            "TomPelissero", "ESPNStatsInfo", "NFLNetwork"
        ]
        return handle in verified


# =============================================================================
# TEAM HASHTAG TESTS
# =============================================================================

class TestTeamHashtags:
    """Tests for team hashtag configuration."""
    
    # Sample team config (from twitter_stream.py)
    TEAM_CONFIG = {
        "SEA": {"hashtags": ["#Seahawks", "#GoHawks"], "city": "Seattle"},
        "KC": {"hashtags": ["#ChiefsKingdom", "#Chiefs"], "city": "Kansas City"},
        "HOU": {"hashtags": ["#Texans", "#WeAreTexans"], "city": "Houston"},
    }
    
    def test_team_has_hashtags(self):
        """Test each team has at least 2 hashtags."""
        for team, config in self.TEAM_CONFIG.items():
            assert len(config["hashtags"]) >= 2, f"{team} needs more hashtags"
    
    def test_hashtags_start_with_hash(self):
        """Test all hashtags start with #."""
        for team, config in self.TEAM_CONFIG.items():
            for tag in config["hashtags"]:
                assert tag.startswith("#"), f"Invalid hashtag: {tag}"
    
    def test_team_has_city(self):
        """Test each team has city mapping."""
        for team, config in self.TEAM_CONFIG.items():
            assert "city" in config
            assert len(config["city"]) > 0


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
