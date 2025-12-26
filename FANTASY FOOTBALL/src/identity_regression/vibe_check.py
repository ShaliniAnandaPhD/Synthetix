"""
VibeCheckScorer - Identity consistency scoring for AI sportscasters.
"""
import json
import os
from typing import Dict, List

def load_archetype_config() -> Dict:
    """Load archetype configuration."""
    config_path = os.path.join(os.path.dirname(__file__), "../../config/sportscaster_archetypes.json")
    with open(config_path) as f:
        return json.load(f)


class VibeCheckScorer:
    """
    Score responses for identity consistency.
    Detects when an agent's personality drifts from its archetype.
    """
    
    INTENSITY_WORDS = ['RIDICULOUS', 'INSANE', 'ABSOLUTELY', 'DOMINANT', 
                       'PATHETIC', 'ELITE', 'LAUGHABLE', 'UNBELIEVABLE',
                       'INCREDIBLE', 'AMAZING', 'TERRIBLE', 'AWFUL']
    
    EVIDENCE_MARKERS = {
        'advanced_stats': ['EPA', 'DVOA', 'yards per attempt', 'efficiency', 'PFF', 
                          'target share', 'completion percentage', 'passer rating'],
        'historical': ['reminds me of', 'back in', 'since 19', 'legacy', 
                      'hall of fame', 'history', 'dynasty', 'era'],
        'film_study': ['tape', 'footwork', 'pre-snap', 'route', 'coverage', 
                      'technique', 'scheme', 'formation'],
        'gut_feel': ['trust me', 'I feel', 'my gut', 'you just know', 
                    'obvious', 'clearly', 'no doubt'],
        'city_pride': ['our', 'we', 'us', 'home', 'kingdom', 'nation', 
                      'faithful', 'believes'],
        'balanced': ['both teams', 'credit to', 'acknowledge', 'fair', 
                    'on the other hand', 'however', 'that said'],
        'contrarian': ['everyone is wrong', 'overrated', 'underrated', 
                      'hot take', 'unpopular opinion'],
        'narrative': ['story', 'journey', 'moment', 'destiny', 'chapter', 
                     'legacy', 'defining'],
        'matchup': ['versus', 'against', 'head to head', 'comparison', 
                   'matchup', 'battle'],
        'emotional': ['love', 'hate', 'passion', 'heart', 'grit', 
                     'fight', 'believe', 'champion']
    }
    
    def __init__(self, archetype: str, archetype_config: Dict):
        self.archetype = archetype
        self.archetype_config = archetype_config
        self.config = archetype_config.get(archetype, {})
        self.signature_phrases = self.config.get("signature_phrases", [])
        self.energy_baseline = self.config.get("energy_baseline", 0.5)
        self.evidence_weights = self.config.get("evidence_weights", {})
    
    def score(self, response: str) -> float:
        """
        Returns 0-1 score for identity fidelity.
        
        Weights:
        - Signature phrase presence: 0.4
        - Energy level match: 0.3
        - Evidence style match: 0.3
        """
        if self.archetype == "neutral":
            return self._score_neutral(response)
        
        phrase_score = self._check_phrases(response)
        energy_score = self._check_energy(response)
        evidence_score = self._check_evidence(response)
        
        return (phrase_score * 0.4) + (energy_score * 0.3) + (evidence_score * 0.3)
    
    def _check_phrases(self, response: str) -> float:
        """Check if signature phrases appear, scaled by archetype."""
        matches = sum(1 for p in self.signature_phrases if p.lower() in response.lower())
        expected_phrases = max(2, len(self.signature_phrases) * 0.5)
        return min(1.0, matches / expected_phrases)
    
    def _check_energy(self, response: str) -> float:
        """Check energy level via exclamations, caps, intensity words."""
        exclamations = response.count('!')
        caps_words = sum(1 for word in response.split() if word.isupper() and len(word) > 2)
        intensity_matches = sum(1 for word in self.INTENSITY_WORDS if word in response.upper())
        
        # Raw energy score (capped at 1.0)
        raw_energy = min(1.0, (exclamations * 0.1) + (caps_words * 0.15) + (intensity_matches * 0.2))
        
        # Penalize deviation from baseline
        deviation = abs(raw_energy - self.energy_baseline)
        return 1.0 - min(1.0, deviation * 2)
    
    def _check_evidence(self, response: str) -> float:
        """Check if evidence style matches archetype preferences."""
        response_lower = response.lower()
        
        # Score each evidence category
        category_scores = {}
        for category, markers in self.EVIDENCE_MARKERS.items():
            matches = sum(1 for m in markers if m.lower() in response_lower)
            category_scores[category] = min(1.0, matches / 2)
        
        # Weight by archetype's preferences
        weighted_score = 0.0
        total_weight = 0.0
        for category, weight in self.evidence_weights.items():
            if category in category_scores:
                weighted_score += category_scores[category] * weight
                total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.5
    
    def _score_against(self, response: str, other_archetype: str) -> float:
        """Check how much response matches a different archetype."""
        other_config = self.archetype_config.get(other_archetype, {})
        other_phrases = other_config.get("signature_phrases", [])
        
        matches = sum(1 for p in other_phrases if p.lower() in response.lower())
        return min(1.0, matches / 2)
    
    def _score_neutral(self, response: str) -> float:
        """
        Neutral archetype: penalize if response matches OTHER archetypes.
        A good neutral response should NOT sound like a homer or hot-take artist.
        """
        max_other_score = max(
            self._score_against(response, arch) 
            for arch in ["hot_take_artist", "homer", "statistician"]
        )
        
        # High score if we DON'T match other archetypes
        neutrality = 1.0 - max_other_score
        
        # Also check for balanced indicators
        balance_score = self._check_phrases(response)
        
        return (neutrality * 0.6) + (balance_score * 0.4)
