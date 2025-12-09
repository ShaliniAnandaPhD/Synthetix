"""
Tempo Engine for managing conversational timing and interruption logic.
"""

import json
import os
import random
from typing import Dict, Any, Optional


class TempoEngine:
    """Engine for controlling conversational tempo and interruption behavior."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the TempoEngine.
        
        Args:
            config_path: Path to the city_profiles.json file.
                        Defaults to config/city_profiles.json relative to project root.
        """
        if config_path is None:
            # Default to config/city_profiles.json from project root
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            config_path = os.path.join(project_root, 'config', 'city_profiles.json')
        
        self.config_path = config_path
        self._profiles = None
    
    def _load_profiles(self) -> Dict[str, Any]:
        """
        Load all city profiles from JSON file.
        
        Returns:
            Dictionary of city profiles.
        """
        if self._profiles is None:
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._profiles = json.load(f)
            except FileNotFoundError:
                raise FileNotFoundError(f"City profiles config not found at: {self.config_path}")
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(f"Invalid JSON in city profiles config: {e.msg}", e.doc, e.pos)
        
        return self._profiles
    
    def _load_city_profile(self, city_name: str) -> Dict[str, Any]:
        """
        Load a specific city's profile.
        
        Args:
            city_name: Name of the city
        
        Returns:
            City profile dictionary.
        
        Raises:
            KeyError: If city not found.
        """
        profiles = self._load_profiles()
        
        if city_name not in profiles:
            available_cities = ', '.join(profiles.keys())
            raise KeyError(
                f"City '{city_name}' not found. Available: {available_cities}"
            )
        
        return profiles[city_name]
    
    def get_delay(self, city_name: str) -> float:
        """
        Calculate response delay for a city in seconds.
        
        This uses base_delay_ms + random variance and converts to seconds.
        
        Args:
            city_name: Name of the city
        
        Returns:
            Delay in seconds (float).
        
        Raises:
            KeyError: If city not found or tempo config missing.
        """
        try:
            profile = self._load_city_profile(city_name)
            tempo = profile.get('tempo', {})
            
            # Get base delay with default fallback
            base_delay_ms = tempo.get('base_delay_ms', 150)
            
            # Get variance with default fallback
            variance_ms = tempo.get('variance_ms', 20)
            
            # Calculate random delay within variance range
            # Using uniform distribution: base ± (variance / 2)
            min_delay_ms = base_delay_ms - (variance_ms / 2)
            max_delay_ms = base_delay_ms + (variance_ms / 2)
            
            delay_ms = random.uniform(min_delay_ms, max_delay_ms)
            
            # Convert to seconds
            delay_seconds = delay_ms / 1000.0
            
            return delay_seconds
            
        except KeyError as e:
            raise KeyError(f"Error getting delay for '{city_name}': {e}")
    
    def check_interruption(self, city_name: str, opponent_confidence: float) -> bool:
        """
        Determine if this city should interrupt based on opponent confidence.
        
        Returns True if opponent_confidence is below the city's interruption threshold.
        
        Args:
            city_name: Name of the city
            opponent_confidence: Opponent's confidence level (0.0 to 1.0)
        
        Returns:
            True if should interrupt, False otherwise.
        
        Raises:
            KeyError: If city not found or interruption config missing.
        """
        try:
            profile = self._load_city_profile(city_name)
            interruption = profile.get('interruption', {})
            
            # Get interruption threshold with default fallback
            threshold = interruption.get('threshold', 0.5)
            
            # Interrupt if opponent confidence is below threshold
            should_interrupt = opponent_confidence < threshold
            
            return should_interrupt
            
        except KeyError as e:
            raise KeyError(f"Error checking interruption for '{city_name}': {e}")
    
    def get_aggression_level(self, city_name: str) -> float:
        """
        Get the aggression level for a city.
        
        Args:
            city_name: Name of the city
        
        Returns:
            Aggression level (0.0 to 1.0).
        """
        try:
            profile = self._load_city_profile(city_name)
            interruption = profile.get('interruption', {})
            return interruption.get('aggression', 0.5)
        except KeyError:
            return 0.5  # Default moderate aggression
    
    def should_back_down(self, city_name: str) -> bool:
        """
        Determine if the city should back down from an argument.
        
        Uses random probability against backs_down_rate.
        
        Args:
            city_name: Name of the city
        
        Returns:
            True if should back down, False otherwise.
        """
        try:
            profile = self._load_city_profile(city_name)
            interruption = profile.get('interruption', {})
            backs_down_rate = interruption.get('backs_down_rate', 0.3)
            
            # Random roll against backs_down_rate
            return random.random() < backs_down_rate
            
        except KeyError:
            return random.random() < 0.3  # Default 30% chance
    
    def get_confidence_threshold(self, city_name: str) -> float:
        """
        Get the confidence threshold for engaging in debate.
        
        Args:
            city_name: Name of the city
        
        Returns:
            Confidence threshold (0.0 to 1.0).
        """
        try:
            profile = self._load_city_profile(city_name)
            tempo = profile.get('tempo', {})
            return tempo.get('confidence_threshold', 0.6)
        except KeyError:
            return 0.6  # Default moderate threshold
