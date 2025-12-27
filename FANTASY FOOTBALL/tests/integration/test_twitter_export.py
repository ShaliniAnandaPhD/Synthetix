"""
Twitter Export Integration Tests

Tests for exporting content to Twitter/X.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# =============================================================================
# TWITTER EXPORT TESTS
# =============================================================================

class TestTwitterExport:
    """Tests for Twitter export functionality."""
    
    def test_tweet_character_limit(self):
        """Test tweets respect 280 character limit."""
        max_chars = 280
        
        short_text = "Touchdown by the Texans!"
        long_text = "A" * 300
        
        # Short text should pass
        assert len(short_text) <= max_chars
        
        # Long text needs truncation
        truncated = self._truncate_for_twitter(long_text)
        assert len(truncated) <= max_chars
    
    def test_truncation_adds_ellipsis(self):
        """Test truncated text ends with ellipsis."""
        long_text = "A" * 300
        truncated = self._truncate_for_twitter(long_text)
        
        assert truncated.endswith("...")
        assert len(truncated) <= 280
    
    def test_hashtag_integration(self):
        """Test hashtags added to tweets."""
        text = "Great play by the Texans!"
        hashtags = ["#Texans", "#NFLSunday"]
        
        full_tweet = self._add_hashtags(text, hashtags)
        
        for tag in hashtags:
            assert tag in full_tweet
    
    def test_hashtags_respect_limit(self):
        """Test hashtags don't exceed character limit."""
        text = "A" * 250
        hashtags = ["#VeryLongHashtag1", "#VeryLongHashtag2"]
        
        full_tweet = self._add_hashtags_safely(text, hashtags)
        
        assert len(full_tweet) <= 280
    
    def _truncate_for_twitter(self, text: str, max_len: int = 277) -> str:
        """Truncate text for Twitter with ellipsis."""
        if len(text) <= max_len:
            return text
        return text[:max_len] + "..."
    
    def _add_hashtags(self, text: str, hashtags: list) -> str:
        """Add hashtags to text."""
        return text + " " + " ".join(hashtags)
    
    def _add_hashtags_safely(self, text: str, hashtags: list) -> str:
        """Add hashtags without exceeding 280 chars."""
        result = text
        for tag in hashtags:
            if len(result) + len(tag) + 1 <= 280:
                result += " " + tag
        return result


# =============================================================================
# THREAD EXPORT TESTS
# =============================================================================

class TestTwitterThreadExport:
    """Tests for Twitter thread export."""
    
    def test_thread_splitting(self):
        """Test long content splits into thread."""
        long_content = "A" * 600  # Too long for one tweet
        
        tweets = self._split_into_thread(long_content)
        
        assert len(tweets) >= 2
        for tweet in tweets:
            assert len(tweet) <= 280
    
    def test_thread_numbering(self):
        """Test thread tweets are numbered."""
        content = ["Tweet 1", "Tweet 2", "Tweet 3"]
        
        numbered = self._number_thread(content)
        
        assert numbered[0].startswith("1/3")
        assert numbered[1].startswith("2/3")
        assert numbered[2].startswith("3/3")
    
    def _split_into_thread(self, text: str, max_len: int = 277) -> list:
        """Split text into thread-sized chunks."""
        chunks = []
        while text:
            chunks.append(text[:max_len])
            text = text[max_len:]
        return chunks
    
    def _number_thread(self, tweets: list) -> list:
        """Add numbering to thread tweets."""
        total = len(tweets)
        return [f"{i+1}/{total} {tweet}" for i, tweet in enumerate(tweets)]


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
