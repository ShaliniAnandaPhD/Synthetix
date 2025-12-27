"""
AI Commentary Unit Tests

Tests for city profiles, archetypes, vibe checking, and multi-agent debates.
"""
import pytest
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.identity_regression import (
    VibeCheckScorer, 
    load_archetype_config,
    load_platinum_traces,
    PlatinumFallbackSystem
)


# =============================================================================
# CITY PROFILES TESTS
# =============================================================================

class TestCityProfiles:
    """Tests for city profile loading and validation."""
    
    def test_city_profiles_loaded(self):
        """Verify city profiles exist and are valid."""
        config_path = os.path.join(
            os.path.dirname(__file__), 
            "../..", 
            "config/city_profiles.json"
        )
        
        # Normalize path
        config_path = os.path.abspath(config_path)
        
        if not os.path.exists(config_path):
            pytest.skip(f"City profiles not found at {config_path}")
        
        with open(config_path) as f:
            profiles = json.load(f)
        
        # Should have profiles
        assert len(profiles) > 0, "No city profiles found"
        
        # Check a known city exists (case insensitive)
        profile_keys = [k.lower() for k in profiles.keys()]
        assert any("houston" in k for k in profile_keys), "Houston profile missing"


# =============================================================================
# ARCHETYPE TESTS
# =============================================================================

class TestArchetypes:
    """Tests for archetype configuration."""
    
    def test_archetype_definitions(self):
        """Verify all 6 archetypes are properly configured."""
        archetypes = load_archetype_config()
        assert len(archetypes) == 6
        
        required = ["statistician", "historian", "hot_take_artist", 
                    "analyst", "homer", "neutral"]
        assert set(archetypes.keys()) == set(required)
    
    def test_archetype_fields(self):
        """Verify each archetype has required fields."""
        archetypes = load_archetype_config()
        
        for name, arch in archetypes.items():
            assert "signature_phrases" in arch, f"{name} missing signature_phrases"
            assert "energy_baseline" in arch, f"{name} missing energy_baseline"
            assert 0.0 <= arch["energy_baseline"] <= 1.0, f"{name} energy_baseline out of range"
            assert len(arch["signature_phrases"]) >= 3, f"{name} needs at least 3 phrases"
    
    def test_archetype_uniqueness(self):
        """Verify archetypes have distinct characteristics."""
        archetypes = load_archetype_config()
        
        energy_levels = [arch["energy_baseline"] for arch in archetypes.values()]
        # Not all archetypes should have the same energy
        assert len(set(energy_levels)) >= 3


# =============================================================================
# VIBE CHECK TESTS
# =============================================================================

class TestVibeCheck:
    """Tests for vibe check scoring."""
    
    @pytest.fixture
    def archetypes(self):
        return load_archetype_config()
    
    def test_homer_authentic_response(self, archetypes):
        """Homer responses with homer phrases should score high."""
        scorer = VibeCheckScorer("homer", archetypes)
        
        authentic = "OUR guys just dominated! This is championship mentality! Nobody believes in us!"
        score = scorer.score(authentic)
        
        assert score > 0.6, f"Authentic homer response scored too low: {score}"
    
    def test_homer_generic_response(self, archetypes):
        """Generic responses should score low for homer archetype."""
        scorer = VibeCheckScorer("homer", archetypes)
        
        generic = "The team played well today. Both sides showed good effort."
        score = scorer.score(generic)
        
        assert score < 0.7, f"Generic response scored too high for homer: {score}"
    
    def test_statistician_data_response(self, archetypes):
        """Statistician responses with data should score high."""
        scorer = VibeCheckScorer("statistician", archetypes)
        
        data_driven = "The numbers don't lie - if you look at the data, the efficiency rating shows EPA per play is high."
        score = scorer.score(data_driven)
        
        assert score > 0.5, f"Data-driven response scored too low: {score}"
    
    def test_hot_take_artist_energy(self, archetypes):
        """Hot take artist with high energy should score high."""
        scorer = VibeCheckScorer("hot_take_artist", archetypes)
        
        hot_take = "Are you KIDDING me?! This is RIDICULOUS! DOMINANT performance and it's not even CLOSE!"
        score = scorer.score(hot_take)
        
        assert score > 0.6, f"Hot take response scored too low: {score}"
    
    def test_neutral_balanced_response(self, archetypes):
        """Neutral responses should score high for neutral archetype."""
        scorer = VibeCheckScorer("neutral", archetypes)
        
        balanced = "Credit to both teams for a hard-fought game. You have to acknowledge the execution on both sides."
        score = scorer.score(balanced)
        
        assert score > 0.6, f"Balanced response scored too low for neutral: {score}"
    
    def test_score_range(self, archetypes):
        """Scores should always be between 0 and 1."""
        scorer = VibeCheckScorer("homer", archetypes)
        
        test_responses = [
            "",
            "Short.",
            "A" * 1000,
            "our OUR OUR!!!!",
            "The data shows efficiency rating EPA per play if you look at the numbers"
        ]
        
        for response in test_responses:
            score = scorer.score(response)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range for: {response[:50]}"


# =============================================================================
# PLATINUM FALLBACK TESTS
# =============================================================================

class TestPlatinumFallback:
    """Tests for platinum fallback system."""
    
    @pytest.fixture
    def fallback(self):
        return PlatinumFallbackSystem()
    
    def test_fallback_system_loads(self, fallback):
        """Verify fallback system initializes with responses."""
        stats = fallback.archive_stats()
        
        assert stats["total_responses"] > 0
        assert stats["cities_covered"] > 0
    
    def test_city_event_fallback(self, fallback):
        """Test specific city + event type returns valid fallback."""
        response = fallback.get_fallback("houston", "touchdown")
        
        assert response is not None
        assert len(response) > 10
    
    def test_generic_fallback(self, fallback):
        """Test generic event type works."""
        response = fallback.get_fallback("houston", "generic")
        
        assert response is not None
        assert len(response) > 10
    
    def test_unknown_city_fallback(self, fallback):
        """Unknown city should still return something."""
        response = fallback.get_fallback("unknown_city", "touchdown")
        
        assert response is not None
        assert len(response) > 10
    
    def test_all_cities_have_fallbacks(self, fallback):
        """All Saturday game cities should have fallbacks."""
        cities = ["houston", "los_angeles", "baltimore", "green_bay"]
        events = ["touchdown", "interception", "generic"]
        
        for city in cities:
            for event in events:
                response = fallback.get_fallback(city, event)
                assert response is not None, f"No fallback for {city}:{event}"
                assert "technical difficulties" not in response.lower(), \
                    f"Emergency fallback used for {city}:{event}"


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
