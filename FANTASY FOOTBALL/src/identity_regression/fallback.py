"""
Platinum fallback system for when generation fails vibe check.
"""
import json
import os
import random
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class PlatinumFallbackSystem:
    """
    3-tier fallback: exact → generic → universal.
    
    Archive Structure:
    {
        "houston:touchdown": ["Response 1", "Response 2", ...],
        "houston:interception": [...],
        "houston:generic": [...],
        ...
    }
    
    Event Types:
    - touchdown, field_goal, interception, fumble
    - sack, big_play, penalty, momentum_shift
    - injury, timeout, challenge, generic
    """
    
    UNIVERSAL_FALLBACK = "That's a significant play that could shift momentum here."
    
    def __init__(self, archive_path: str = None):
        if archive_path is None:
            archive_path = os.path.join(
                os.path.dirname(__file__), 
                "../../config/platinum_traces.json"
            )
        
        try:
            with open(archive_path) as f:
                self.archive = json.load(f)
            logger.info(f"Loaded platinum archive: {len(self.archive)} keys")
        except FileNotFoundError:
            logger.error(f"Platinum archive not found at {archive_path}")
            self.archive = {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in platinum archive: {e}")
            self.archive = {}
    
    def get_fallback(self, city: str, event_type: str) -> str:
        """
        Returns curated response from archive.
        3-tier fallback: exact → generic → universal.
        
        Args:
            city: City name (e.g., "Houston", "los_angeles")
            event_type: Event type (e.g., "touchdown", "interception")
        
        Returns:
            Fallback response string
        """
        city_key = city.lower().replace(" ", "_")
        
        # Tier 1: Exact match
        key = f"{city_key}:{event_type}"
        if key in self.archive and self.archive[key]:
            logger.debug(f"Platinum fallback: exact match for {key}")
            return random.choice(self.archive[key])
        
        # Tier 2: Generic for city
        generic_key = f"{city_key}:generic"
        if generic_key in self.archive and self.archive[generic_key]:
            logger.warning(f"Platinum fallback: no match for {key}, using {generic_key}")
            return random.choice(self.archive[generic_key])
        
        # Tier 3: Universal fallback
        logger.error(f"Platinum fallback: no match for {key} or {generic_key}, using universal")
        return self.UNIVERSAL_FALLBACK
    
    def archive_stats(self) -> Dict:
        """Return stats about archive coverage."""
        if not self.archive:
            return {"total_responses": 0, "cities_covered": 0, "event_types": []}
        
        # Parse keys to extract cities and event types
        cities = set()
        events = set()
        
        for key in self.archive.keys():
            if ":" in key:
                city, event = key.split(":", 1)
                cities.add(city)
                events.add(event)
        
        return {
            "total_responses": sum(len(v) for v in self.archive.values() if isinstance(v, list)),
            "cities_covered": len(cities),
            "cities": list(cities),
            "event_types": list(events)
        }
